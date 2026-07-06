// Keyboard driving for the quiz loop. Number keys pick from whatever
// choice set is on screen (answer buttons before the reveal, the FSRS
// bar after), space/enter is the primary action, and r replays audio.
(() => {
  const quiz = () => document.getElementById("quiz")
  const busy = () => quiz()?.classList.contains("htmx-request")
  const visible = (el) => el && el.offsetParent !== null

  // A key that changes what's on screen makes the next key mean
  // something new; a brief cooldown keeps a double-press from
  // answering *and* rating in one go.
  let lastAction = 0
  const acted = () => {
    const now = Date.now()
    if (now - lastAction < 350) return false
    lastAction = now
    return true
  }

  const feedbackButtons = () => {
    const bar = document.querySelector(".feedback")
    return visible(bar) ? [...bar.querySelectorAll("button")] : null
  }

  const answerButtons = () => {
    const buttons = [...document.querySelectorAll("#quiz .answer")]
    return buttons.some(visible) ? buttons : null
  }

  // Reveal/Learn buttons sit directly under the card; the Start
  // button directly under #quiz. Form and answer buttons don't.
  const primaryButton = () => {
    const btn =
      document.querySelector("#quiz article > button") ||
      document.querySelector("#quiz > button")
    return visible(btn) && !btn.disabled ? btn : null
  }

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return
    const t = e.target
    if (t.matches?.("input, textarea, select") || t.isContentEditable) return
    if (busy()) return

    if (e.key === "r" || e.key === "R") {
      document.getElementById("pronounce")?.play().catch(() => {})
      e.preventDefault()
      return
    }

    if (e.key === " " || e.key === "Enter") {
      // With the rating bar up, space means Good, as in Anki.
      const target = primaryButton() || (feedbackButtons() || [])[2]
      if (target && acted()) {
        target.click()
        e.preventDefault()
      }
      return
    }

    if (/^[1-9]$/.test(e.key)) {
      const group = feedbackButtons() || answerButtons()
      const btn = group?.[Number(e.key) - 1]
      if (btn && !btn.disabled && visible(btn) && acted()) {
        btn.click()
        e.preventDefault()
      }
    }
  })

  // Typing quizzes: put the cursor in the box as the card arrives,
  // and let go of it on submit so the rating keys work right after.
  document.body.addEventListener("htmx:afterSwap", (e) => {
    if (e.target.id === "quiz") {
      document.querySelector("#quiz input[type=text]")?.focus()
      lastAction = Date.now()
    }
  })

  document.addEventListener(
    "submit",
    (e) => e.target.querySelector("input[type=text]")?.blur(),
    true
  )
})()
