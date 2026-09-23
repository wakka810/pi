import { describe, expect, it } from "vitest";
import { convertResponsesMessages, getResponsesReasoningEffortState } from "../src/api/openai-responses-shared.ts";
import type { AssistantMessage, Message, Model } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const TOOL_CALL_PROVIDERS = new Set(["openai"]);

function model(): Model<"openai-responses"> {
	return {
		id: "gpt-6-luna",
		name: "GPT-6 Luna",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://api.openai.com/v1",
		reasoning: true,
		thinkingLevelMap: { off: "none", low: "low", medium: "medium", high: "high", xhigh: "xhigh", max: "max" },
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 272000,
		maxTokens: 128000,
		compat: { supportsReasoningEffortUpdates: true },
	};
}

function assistant(level: string, text: string, timestamp: number): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "openai-responses",
		provider: "openai",
		model: "gpt-6-luna",
		providerThinkingLevel: level,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp,
	};
}

function user(text: string, timestamp: number): Message {
	return { role: "user", content: text, timestamp };
}

function wire(messages: Message[], baseline: string) {
	return convertResponsesMessages(model(), normalizeContext({ messages }), TOOL_CALL_PROVIDERS, {
		reasoningEffortBaseline: baseline,
	});
}

describe("OpenAI cache-preserving reasoning effort updates", () => {
	it("pins the request baseline to the first managed assistant effort", () => {
		const messages = [user("one", 1), assistant("low", "first", 2), user("two", 3), assistant("high", "second", 4)];
		expect(getResponsesReasoningEffortState(model(), messages, "high")).toEqual({
			baselineEffort: "low",
			effectiveEffort: "high",
		});
	});

	it("replays an effort change immediately before the assistant generated under it", () => {
		const messages = [
			user("one", 1),
			assistant("low", "first", 2),
			user("two", 3),
			assistant("high", "second", 4),
			user("three", 5),
		];
		const input = wire(messages, "low");
		const updateIndex = input.findIndex((item) => (item as { type?: string }).type === "configuration_update");
		const secondAssistantIndex = input.findIndex(
			(item) =>
				(item as { type?: string; role?: string; content?: Array<{ text?: string }> }).type === "message" &&
				(item as { role?: string }).role === "assistant" &&
				(item as { content?: Array<{ text?: string }> }).content?.[0]?.text === "second",
		);

		expect(updateIndex).toBeGreaterThan(-1);
		expect(secondAssistantIndex).toBe(updateIndex + 1);
		expect(input.filter((item) => (item as { type?: string }).type === "configuration_update")).toEqual([
			{ type: "configuration_update", reasoning: { effort: "high" } },
		]);
	});

	it("does not duplicate an unchanged historical effort", () => {
		const messages = [user("one", 1), assistant("low", "first", 2), user("two", 3), assistant("low", "second", 4)];
		const input = wire(messages, "low");
		expect(input.some((item) => (item as { type?: string }).type === "configuration_update")).toBe(false);
	});

	it("replays multiple effort changes in their original order", () => {
		const messages = [
			user("one", 1),
			assistant("low", "first", 2),
			user("two", 3),
			assistant("high", "second", 4),
			user("three", 5),
			assistant("low", "third", 6),
		];
		const input = wire(messages, "low");
		expect(
			input
				.filter((item) => (item as { type?: string }).type === "configuration_update")
				.map((item) => (item as { reasoning?: { effort?: string } }).reasoning?.effort),
		).toEqual(["high", "low"]);
	});

	it("starts a new effort epoch after a foreign-model assistant", () => {
		const foreign = { ...assistant("low", "foreign", 4), model: "gpt-6-sol" };
		const messages = [user("one", 1), assistant("low", "first", 2), user("handoff", 3), foreign, user("back", 5)];
		expect(getResponsesReasoningEffortState(model(), messages, "high")).toEqual({
			baselineEffort: "high",
			effectiveEffort: "high",
		});
		expect(wire(messages, "high").some((item) => (item as { type?: string }).type === "configuration_update")).toBe(
			false,
		);
	});
});
