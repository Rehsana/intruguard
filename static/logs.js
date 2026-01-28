document.addEventListener("DOMContentLoaded", async () => {
  try {
    const response = await fetch("/api/logs");
    const logs = await response.json();

    const logList = document.getElementById("logList");
    logs.forEach(log => {
      const li = document.createElement("li");
      li.textContent = `${log.timestamp} - ${log.message}`;
      logList.appendChild(li);
    });
  } catch (err) {
    console.error(err);
  }
});
