// Solve the login form's proof-of-work challenge before submitting.
// Batched so WebCrypto's per-digest promise overhead doesn't dominate.
const count_leading_zero_bits = (bytes) => {
  let bits = 0
  for (const byte of bytes) {
    if (byte === 0) {
      bits += 8
      continue
    }
    return bits + Math.clz32(byte) - 24
  }
  return bits
}

const solve_pow = async (nonce, difficulty) => {
  const encoder = new TextEncoder()
  const BATCH = 128
  for (let base = 0; ; base += BATCH) {
    const digests = await Promise.all(
      Array.from({ length: BATCH }, (_, offset) =>
        crypto.subtle.digest(
          "SHA-256",
          encoder.encode(nonce + ":" + (base + offset))
        )
      )
    )
    for (let offset = 0; offset < BATCH; offset++) {
      if (count_leading_zero_bits(new Uint8Array(digests[offset])) >= difficulty) {
        return String(base + offset)
      }
    }
  }
}

document.querySelectorAll("form[data-pow]").forEach((form) => {
  form.addEventListener("submit", async (event) => {
    event.preventDefault()

    const button = form.querySelector("button[type=submit]")
    button.disabled = true
    button.textContent = "Proving you're not a robot…"

    form.elements.pow_solution.value = await solve_pow(
      form.elements.pow_nonce.value,
      parseInt(form.elements.pow_difficulty.value, 10)
    )

    // form.submit() skips the submit event, so this can't re-enter us.
    form.submit()
  })
})
