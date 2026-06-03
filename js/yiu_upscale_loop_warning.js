import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let lastKey = null;
let lastAt = 0;

function addToast(summary, detail) {
	const toast = app?.extensionManager?.toast?.add;
	if (typeof toast === "function") {
		toast({
			severity: "warn",
			summary,
			detail,
			life: 9000,
		});
	}
}

app.registerExtension({
	name: "yiu.upscale_loop.preview_warning",
	async setup() {
		api.addEventListener("yiu_upscale_loop_preview_warning", (event) => {
			const detail = event?.detail ?? {};
			const width = Number(detail.width);
			const height = Number(detail.height);
			const pixels = Number(detail.pixels);
			const message = detail.message_zh || detail.message_en || "";
			const key = `${width}x${height}:${pixels}:${message}`;
			const now = Date.now();
			if (key === lastKey && now - lastAt < 5000) return;
			lastKey = key;
			lastAt = now;
			if (message) {
				console.warn(`[yiu_nodes] ${message}`);
				addToast("Upscale Loop Warning", message);
			} else if (Number.isFinite(width) && Number.isFinite(height)) {
				const fallback = `Output image is ${width}x${height}. ComfyUI preview nodes may fail to display it. Use a Save node, or check ComfyUI's temp folder.`;
				console.warn(`[yiu_nodes] ${fallback}`);
				addToast("Upscale Loop Warning", fallback);
			}
		});
	},
});
