document.addEventListener("DOMContentLoaded", async () => {
  try {
    // Example fetch to backend for dashboard statistics
    const res = await fetch("/api/dashboard-data");
    const data = await res.json();

    if (data) {
      document.getElementById("totalNetwork").textContent = data.totalNetwork;
      document.getElementById("totalWeb").textContent = data.totalWeb;
    }
  } catch (err) {
    console.error(err);
  }
});
