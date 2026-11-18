const API_URL = new URLSearchParams(location.search).get("api") || "";

const form = document.getElementById("form");
const btn = document.getElementById("btn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");

function show(el, on) { el.classList.toggle("hidden", !on); }

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  show(errorBox, false);
  show(resultBox, false);

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const data = {};
  for (const [k, v] of new FormData(form).entries()) data[k] = Number(v);

  btn.disabled = true;
  show(loading, true);
  try {
    const res = await fetch(API_URL + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      let msg = "Request failed (" + res.status + ").";
      if (Array.isArray(body.detail)) {
        msg = body.detail.map((d) => d.loc.slice(1).join(".") + ": " + d.msg).join("; ");
      } else if (body.detail) {
        msg = body.detail;
      }
      throw new Error(msg);
    }
    const no = body.probability.no_heart_disease * 100;
    const yes = body.probability.heart_disease * 100;
    document.getElementById("result-text").textContent = body.result;
    document.getElementById("pct-no").textContent = no.toFixed(1) + "%";
    document.getElementById("pct-yes").textContent = yes.toFixed(1) + "%";
    document.getElementById("bar-no").style.width = no + "%";
    document.getElementById("bar-yes").style.width = yes + "%";
    resultBox.className = "result " + (body.prediction === 1 ? "pos" : "neg");
    show(resultBox, true);
  } catch (err) {
    const offline = err instanceof TypeError;
    errorBox.textContent = offline
      ? "Cannot reach the backend. Check that the server is running (free hosts can take about a minute to wake up) and try again."
      : "Error: " + err.message;
    show(errorBox, true);
  } finally {
    btn.disabled = false;
    show(loading, false);
  }
});
