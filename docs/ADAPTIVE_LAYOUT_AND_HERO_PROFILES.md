# Adaptive Layout and Hero Profiles v1

This route separates device geometry from hero skill behavior. Device calibration locates the game
content box and applies independent normalized transforms to the joystick, combat, minimap, and
purchase groups. It does not use skill-icon templates. Hero profiles specify only how the fixed
`skill1`/`skill2`/`skill3` slots behave: tap, directional drag, charge release, targeted tap, or
disabled.

The public examples are intentionally non-executable. A local adaptive layout binds an exact build
identity, reference-layout hash, content box, group confidence, and reprojection error. It is
accepted only after five stable frames, per-group confidence at least 0.90, and reprojection error
at most 1.5% of the short content edge. A local hero profile is required for skill input; an unknown
hero can only move, stop, basic attack, and use deterministic purchase/safety functions.

Skill readiness is hero-independent at the layout layer: the runtime compares each button against
its own session baseline and watches temporal cooldown overlays. It never compares a hero's icon to
another hero. Skill semantics and any drag/hold behavior remain profile-bound and fail closed.

This first implementation provides the validated geometry, hash, and profile contracts. It does
not change the existing live execution path until read-only per-device calibration passes.
