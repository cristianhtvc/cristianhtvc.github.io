const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const pickFile = document.querySelector("#pick-file");
const replaceInput = document.querySelector("#replace-input");
const dropZone = document.querySelector("#drop-zone");
const fileName = document.querySelector("#file-name");
const reviewStatus = document.querySelector("#review-status");
const targetStatus = document.querySelector("#target-status");
const publishStatus = document.querySelector("#publish-status");
const messageList = document.querySelector("#message-list");
const successPanel = document.querySelector("#success-panel");
const savedPath = document.querySelector("#saved-path");
const pagePath = document.querySelector("#page-path");
const dialog = document.querySelector("#notice-dialog");
const dialogTitle = document.querySelector("#dialog-title");
const dialogBody = document.querySelector("#dialog-body");
const dialogClose = document.querySelector("#dialog-close");

let selectedFile = null;

function setMessages(messages) {
  messageList.innerHTML = "";
  for (const item of messages) {
    const element = document.createElement("li");
    element.className = item.type;
    element.textContent = item.text;
    messageList.appendChild(element);
  }
}

function showNotice(title, body) {
  dialogTitle.textContent = title;
  dialogBody.textContent = body;
  if (typeof dialog.showModal === "function") {
    dialog.showModal();
  } else {
    alert(`${title}\n\n${body}`);
  }
}

function resetSuccess() {
  successPanel.hidden = true;
  savedPath.textContent = "";
  pagePath.textContent = "";
}

function setBusy(isBusy) {
  pickFile.disabled = isBusy;
  replaceInput.disabled = isBusy;
}

function chooseFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = file.name;
  reviewStatus.textContent = "已选择";
  targetStatus.textContent = "等待审查";
  publishStatus.textContent = "未开始";
  resetSuccess();
  setMessages([{ type: "success", text: `已选择 ${file.name}，准备上传。` }]);
  uploadSelectedFile();
}

async function uploadSelectedFile() {
  if (!selectedFile) {
    showNotice("请选择文件", "请先选择一个 Markdown `.md` 文件。");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  if (replaceInput.checked) {
    formData.append("replace", "true");
  }

  setBusy(true);
  reviewStatus.textContent = "审查中";
  targetStatus.textContent = "自动推断";
  publishStatus.textContent = "等待审查";
  resetSuccess();
  setMessages([{ type: "success", text: "正在审查 Markdown 结构。" }]);

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!payload.ok) {
      const review = payload.review;
      const errors = review?.errors || [payload.error || "上传失败。"];
      const warnings = review?.warnings || [];
      reviewStatus.textContent = "未通过";
      publishStatus.textContent = "已拒绝";
      setMessages([
        ...errors.map((text) => ({ type: "error", text })),
        ...warnings.map((text) => ({ type: "warn", text })),
      ]);
      showNotice("内容审查未通过", errors.slice(0, 3).join("；"));
      return;
    }

    const result = payload.result;
    reviewStatus.textContent = "已通过";
    targetStatus.textContent = `${result.venue.toUpperCase()} ${result.year}`;
    publishStatus.textContent = "已发布";
    savedPath.textContent = result.saved_path;
    pagePath.textContent = result.page_path;
    successPanel.hidden = false;

    const warnings = result.review.warnings || [];
    setMessages([
      { type: "success", text: `上传成功：${result.title}` },
      ...warnings.map((text) => ({ type: "warn", text })),
    ]);
    showNotice("上传成功", "内容已通过审查，并完成本地站点重建与发布。");
  } catch (error) {
    reviewStatus.textContent = "异常";
    publishStatus.textContent = "未发布";
    setMessages([{ type: "error", text: error.message || "上传请求失败。" }]);
    showNotice("上传异常", error.message || "上传请求失败。");
  } finally {
    setBusy(false);
  }
}

pickFile.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  chooseFile(fileInput.files[0]);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  uploadSelectedFile();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  chooseFile(event.dataTransfer.files[0]);
});

dialogClose.addEventListener("click", () => {
  dialog.close();
});
