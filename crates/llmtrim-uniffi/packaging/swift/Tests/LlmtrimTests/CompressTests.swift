import XCTest

@testable import Llmtrim

final class CompressTests: XCTestCase {
    func testCompressProjectsFields() throws {
        let req = #"{"model":"gpt-4o","messages":[{"role":"user","content":"hello world"}],"max_tokens":5}"#
        let out = try compress(input: req, provider: .openAi, preset: "safe")
        XCTAssertEqual(out.provider, "openai")
        XCTAssertEqual(out.model, "gpt-4o")
        XCTAssertTrue(out.tokenizerExact)
        XCTAssertGreaterThan(out.inputTokensBefore, 0)
    }

    func testAutoDetectProvider() throws {
        let req = #"{"system":"s","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#
        let out = try compress(input: req, provider: nil, preset: "safe")
        XCTAssertEqual(out.provider, "anthropic")
    }

    func testUnknownPresetThrows() {
        XCTAssertThrowsError(try compress(input: "{}", provider: .openAi, preset: "no-such-preset")) { error in
            guard case LlmtrimError.UnknownPreset = error else {
                return XCTFail("expected UnknownPreset, got \(error)")
            }
        }
    }
}
