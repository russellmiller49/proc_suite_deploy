/* ui/static/phi_redactor/linking.js
 *
 * Linking/Normalization layer for ATLAS Review UI.
 * Goals:
 *  - canonical keys (stations/specimens/etc.)
 *  - merge detailed primary rows + complete fallback rows
 *  - mark derived rows with provenance flags
 *  - resolve evidence for parent paths / arrays / aliases
 *
 * No external deps.
 */

(function () {
  "use strict";

  // ----------------------------
  // Generic string/key utilities
  // ----------------------------

  function normalizeWhitespaceCase(s) {
    if (s == null) return "";
    return String(s).replace(/\s+/g, " ").trim();
  }

  function stableKeyFromParts() {
    // Deterministic join; keeps keys readable.
    // NOTE: keep this stable.
    var parts = Array.prototype.slice
      .call(arguments)
      .map(function (p) {
        return normalizeWhitespaceCase(p);
      })
      .filter(function (p) {
        return p.length > 0;
      });
    return parts.join("|").toLowerCase();
  }

  // ----------------------------
  // Station token normalization
  // ----------------------------

  function normalizeTokenStation(token) {
    // Canonicalize common station spellings:
    //  - "11Rs" -> "11RS"
    //  - "4 l"  -> "4L"
    //  - "11 rs"-> "11RS"
    if (token == null) return "";
    var t = normalizeWhitespaceCase(token);
    t = t.replace(/[^0-9a-zA-Z]/g, "");
    return t.toUpperCase();
  }

  // ----------------------------
  // Merge helper: primary + fallback
  // ----------------------------

  function shallowCopy(obj) {
    if (obj == null || typeof obj !== "object") return obj;
    if (Array.isArray(obj)) return obj.slice();
    var out = {};
    Object.keys(obj).forEach(function (k) {
      out[k] = obj[k];
    });
    return out;
  }

  function withProvenance(row, prov) {
    var r = shallowCopy(row) || {};
    Object.keys(prov || {}).forEach(function (k) {
      r[k] = prov[k];
    });
    return r;
  }

  function defaultMergeRow(primaryRow, fallbackRow) {
    // Conservative shallow merge:
    // - do not overwrite primary's non-empty fields
    // - fill only empty/undefined/null fields from fallback
    var p = primaryRow ? shallowCopy(primaryRow) : {};
    var f = fallbackRow ? fallbackRow : {};
    var out = shallowCopy(p);
    Object.keys(f).forEach(function (k) {
      if (String(k || "").indexOf("__") === 0) return;
      var pv = out[k];
      var fv = f[k];
      if (pv == null || pv === "") out[k] = fv;
    });
    return out;
  }

  function mergeEntities(opts) {
    var primaryRows = Array.isArray(opts && opts.primaryRows) ? opts.primaryRows : [];
    var fallbackRows = Array.isArray(opts && opts.fallbackRows) ? opts.fallbackRows : [];
    var getKey = typeof (opts && opts.getKey) === "function" ? opts.getKey : function () { return ""; };
    var mergeRow = typeof (opts && opts.mergeRow) === "function" ? opts.mergeRow : defaultMergeRow;
    var preserveOrder = (opts && opts.preserveOrder) || "primaryThenMissingFallback";

    if (preserveOrder !== "primaryThenMissingFallback") {
      throw new Error('mergeEntities: unsupported preserveOrder="' + preserveOrder + '"');
    }

    var out = [];
    var seen = new Set();

    var primaryByKey = new Map();
    for (var i = 0; i < primaryRows.length; i += 1) {
      var row = primaryRows[i];
      var key = String(getKey(row, i, "primary") || "");
      if (!key) continue;
      if (!primaryByKey.has(key)) primaryByKey.set(key, { row: row, idx: i });
    }

    // Emit primary rows first (in original order)
    for (var i2 = 0; i2 < primaryRows.length; i2 += 1) {
      var prow = primaryRows[i2];
      var pkey = String(getKey(prow, i2, "primary") || "");
      if (!pkey) {
        out.push(
          withProvenance(prow, { __derived: false, __source: "primary", __entityKey: "" })
        );
        continue;
      }
      if (seen.has(pkey)) {
        out.push(
          withProvenance(prow, {
            __derived: false,
            __source: "primary",
            __entityKey: pkey,
            __duplicateKey: true,
          })
        );
        continue;
      }
      seen.add(pkey);
      out.push(withProvenance(prow, { __derived: false, __source: "primary", __entityKey: pkey }));
    }

    // Append missing fallback rows (in fallback order)
    for (var j = 0; j < fallbackRows.length; j += 1) {
      var fb = fallbackRows[j];
      var fkey = String(getKey(fb, j, "fallback") || "");
      if (!fkey) continue;

      if (seen.has(fkey)) {
        // If primary exists, fill blanks into the first canonical primary row.
        var prim = primaryByKey.get(fkey);
        if (prim) {
          for (var k2 = 0; k2 < out.length; k2 += 1) {
            var outRow = out[k2];
            if (
              outRow &&
              outRow.__entityKey === fkey &&
              outRow.__source === "primary" &&
              !outRow.__duplicateKey
            ) {
              out[k2] = withProvenance(mergeRow(outRow, fb, { key: fkey }), {
                __derived: false,
                __source: "merged",
                __entityKey: fkey,
              });
              break;
            }
          }
        }
        continue;
      }

      seen.add(fkey);
      var merged = mergeRow(null, fb, { key: fkey });
      out.push(withProvenance(merged, { __derived: true, __source: "fallback", __entityKey: fkey }));
    }

    return out;
  }

  // ----------------------------
  // Evidence resolution
  // ----------------------------

  function spanStart(span) {
    if (!span) return null;
    var s = span.start;
    if (typeof s === "number") return s;
    if (Array.isArray(span.span) && typeof span.span[0] === "number") return span.span[0];
    return null;
  }

  function spanEnd(span) {
    if (!span) return null;
    var e = span.end;
    if (typeof e === "number") return e;
    if (Array.isArray(span.span) && typeof span.span[1] === "number") return span.span[1];
    return null;
  }

  function pushSpans(spans, out, seen) {
    if (!spans) return;
    var arr = Array.isArray(spans) ? spans : [spans];
    for (var i = 0; i < arr.length; i += 1) {
      var s = arr[i];
      if (!s) continue;
      var key = stableKeyFromParts(
        spanStart(s),
        spanEnd(s),
        s.text || s.quote || s.snippet,
        s.source
      );
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(s);
    }
  }

  function collectEvidenceForPrefix(evidence, prefix, out, seen) {
    if (!prefix) return;
    pushSpans(evidence[prefix], out, seen);

    var dotPrefix = prefix + ".";
    var bracketPrefix = prefix + "[";
    var keys = Object.keys(evidence || {});
    for (var i = 0; i < keys.length; i += 1) {
      var k = keys[i];
      if (k === prefix) continue;
      if (k.indexOf(dotPrefix) === 0 || k.indexOf(bracketPrefix) === 0) {
        pushSpans(evidence[k], out, seen);
      }
    }
  }

  function resolveEvidenceForPath(evidenceMap, path, opts) {
    var evidence = evidenceMap && typeof evidenceMap === "object" ? evidenceMap : {};
    var aliases = opts && opts.aliases ? opts.aliases : null;

    var out = [];
    var seen = new Set();

    var raw = normalizeWhitespaceCase(path);
    if (!raw) return out;

    var candidateRoots = [];
    candidateRoots.push(raw);
    if (raw.indexOf("registry.") === 0) candidateRoots.push(raw.slice("registry.".length));

    if (aliases && aliases[raw] && Array.isArray(aliases[raw])) {
      for (var i = 0; i < aliases[raw].length; i += 1) {
        var ap = normalizeWhitespaceCase(aliases[raw][i]);
        if (!ap) continue;
        candidateRoots.push(ap);
        if (ap.indexOf("registry.") === 0) candidateRoots.push(ap.slice("registry.".length));
      }
    }

    for (var j = 0; j < candidateRoots.length; j += 1) {
      collectEvidenceForPrefix(evidence, candidateRoots[j], out, seen);
    }

    out.sort(function (a, b) {
      var as = spanStart(a);
      var bs = spanStart(b);
      if (typeof as === "number" && typeof bs === "number" && as !== bs) return as - bs;
      var ae = spanEnd(a);
      var be = spanEnd(b);
      if (typeof ae === "number" && typeof be === "number" && ae !== be) return ae - be;
      return normalizeWhitespaceCase(a.text || a.quote || "").localeCompare(normalizeWhitespaceCase(b.text || b.quote || ""));
    });

    return out;
  }

  // ----------------------------
  // Export to window namespace
  // ----------------------------

  window.ATLASLinking = {
    normalizeWhitespaceCase: normalizeWhitespaceCase,
    stableKeyFromParts: stableKeyFromParts,
    normalizeTokenStation: normalizeTokenStation,
    mergeEntities: mergeEntities,
    resolveEvidenceForPath: resolveEvidenceForPath,
  };
})();

