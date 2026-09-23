import { describe, expect, it } from "vitest";
import { clampOpenAIPromptCacheKey, OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH } from "../src/api/openai-prompt-cache.ts";

describe("OpenAI prompt cache key normalization", () => {
	it("preserves short keys", () => {
		expect(clampOpenAIPromptCacheKey("session-123")).toBe("session-123");
	});

	it("keeps long keys within the API limit without colliding on a shared prefix", () => {
		const prefix = "x".repeat(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
		const first = clampOpenAIPromptCacheKey(`${prefix}-first`)!;
		const second = clampOpenAIPromptCacheKey(`${prefix}-second`)!;

		expect(Array.from(first)).toHaveLength(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
		expect(Array.from(second)).toHaveLength(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
		expect(first).not.toBe(second);
		expect(first).not.toBe(prefix);
	});
});
