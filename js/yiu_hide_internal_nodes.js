import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const INTERNAL_NODE_TYPES = new Set([
	"__y_i_u__while_loop_start",
	"__y_i_u__while_loop_end",
	"__y_i_u__math_int",
	"__y_i_u__compare",
	"__y_i_u__tiling_meta",
	"__y_i_u__image_tile",
	"__y_i_u__image_untile",
]);

function removeFromArrayRegistry(registry, typeName) {
	if (!registry || typeof registry !== "object") return;
	for (const key of Object.keys(registry)) {
		const value = registry[key];
		if (!Array.isArray(value)) continue;
		registry[key] = value.filter((item) => {
			const itemType = item?.type ?? item?.comfyClass ?? item?.nodeData?.name;
			return itemType !== typeName;
		});
	}
}

function unregisterFrontendNode(typeName) {
	try {
		delete LiteGraph.registered_node_types?.[typeName];
		delete LiteGraph.Nodes?.[typeName];
		removeFromArrayRegistry(LiteGraph.node_types_by_file, typeName);
	} catch (error) {
		console.warn("[yiu_nodes] failed to hide internal node:", typeName, error);
	}
}

function stripInternalNodeDefs(defs) {
	if (!defs || typeof defs !== "object") return defs;
	for (const typeName of INTERNAL_NODE_TYPES) {
		delete defs[typeName];
	}
	return defs;
}

app.registerExtension({
	name: "yiu.hide_internal_nodes",
	async setup() {
		if (!api.__yiuHideInternalNodesPatched && typeof api.getNodeDefs === "function") {
			const originalGetNodeDefs = api.getNodeDefs.bind(api);
			api.getNodeDefs = async function () {
				const defs = await originalGetNodeDefs();
				return stripInternalNodeDefs(defs);
			};
			api.__yiuHideInternalNodesPatched = true;
		}

		requestAnimationFrame(() => {
			for (const typeName of INTERNAL_NODE_TYPES) {
				unregisterFrontendNode(typeName);
			}
		});
	},

	async beforeRegisterNodeDef(nodeType, nodeData) {
		const typeName = nodeType?.comfyClass ?? nodeType?.type ?? nodeData?.name;
		if (!INTERNAL_NODE_TYPES.has(typeName)) return;

		if (nodeData) {
			nodeData.category = "_yiu_hidden";
		}
		if (nodeType) {
			nodeType.category = "_yiu_hidden";
		}

		requestAnimationFrame(() => unregisterFrontendNode(typeName));
	},
});
