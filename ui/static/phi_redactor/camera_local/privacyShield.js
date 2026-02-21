function hasFunction(value) {
  return typeof value === "function";
}

export function initPrivacyShield(opts = {}) {
  const runtimeDoc = opts.document || globalThis.document;
  const runtimeWin = opts.window || globalThis.window;
  const shieldEl = opts.shieldEl || runtimeDoc?.getElementById?.("privacyShield");

  if (!runtimeDoc || !runtimeWin || !shieldEl) {
    return () => {};
  }

  const onBackground = hasFunction(opts.onBackground) ? opts.onBackground : () => {};
  const onResumeRequested = hasFunction(opts.onResumeRequested) ? opts.onResumeRequested : () => {};
  const shouldActivate = hasFunction(opts.shouldActivate) ? opts.shouldActivate : () => true;

  let awaitingResumeTap = false;

  const isShieldEnabled = () => {
    try {
      return Boolean(shouldActivate());
    } catch {
      return false;
    }
  };

  const activate = (reason) => {
    if (!isShieldEnabled()) return;
    shieldEl.classList.add("active");
    shieldEl.setAttribute("aria-hidden", "false");
    if (!awaitingResumeTap) {
      awaitingResumeTap = true;
      onBackground(reason);
      return;
    }
    // If already shielded, still notify so camera/OCR stop calls remain best-effort.
    onBackground(reason);
  };

  const deactivate = () => {
    shieldEl.classList.remove("active");
    shieldEl.setAttribute("aria-hidden", "true");
  };

  const handleVisibility = () => {
    if (runtimeDoc.hidden) {
      activate("visibility_hidden");
      return;
    }

    if (awaitingResumeTap) {
      if (isShieldEnabled()) {
        shieldEl.classList.add("active");
        shieldEl.setAttribute("aria-hidden", "false");
      } else {
        awaitingResumeTap = false;
        deactivate();
      }
    }
  };

  const handlePageHide = () => {
    if (!isShieldEnabled()) return;
    activate("pagehide");
  };

  const handleBlur = () => {
    if (runtimeDoc.hidden) return;
    if (!isShieldEnabled()) return;
    activate("blur");
  };

  const handleResumeTap = () => {
    if (!awaitingResumeTap) return;
    awaitingResumeTap = false;
    deactivate();
    onResumeRequested();
  };

  const handleResumeKey = (event) => {
    const key = String(event?.key || "");
    if (key !== "Enter" && key !== " " && key !== "Spacebar") return;
    event.preventDefault();
    handleResumeTap();
  };

  runtimeDoc.addEventListener("visibilitychange", handleVisibility);
  runtimeWin.addEventListener("pagehide", handlePageHide);
  runtimeWin.addEventListener("blur", handleBlur);
  shieldEl.addEventListener("click", handleResumeTap);
  shieldEl.addEventListener("keydown", handleResumeKey);

  shieldEl.setAttribute("aria-hidden", "true");

  return () => {
    runtimeDoc.removeEventListener("visibilitychange", handleVisibility);
    runtimeWin.removeEventListener("pagehide", handlePageHide);
    runtimeWin.removeEventListener("blur", handleBlur);
    shieldEl.removeEventListener("click", handleResumeTap);
    shieldEl.removeEventListener("keydown", handleResumeKey);
  };
}
