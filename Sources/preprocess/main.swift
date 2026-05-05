import Domain
import Foundation

// Phase 2 — preprocessor: references.json.gz → references.bin
//
// Pipeline (offline, runs outside the runtime container):
//   1. Read resources/references.json.gz, capture SHA256.
//   2. Stream-decompress and parse the top-level JSON array of
//      `{"vector":[14 nums], "label":"fraud"|"legit"}`.
//   3. Quantize each vector to 16-lane Int16 (scale=8192) via Domain.Vectorizer.
//   4. Write resources/references.bin (header + labels + orig_ids + i16 SoA).
//   5. Validate count == 3_000_000 and round-trip a sample.

@main
struct Preprocess {
    static func main() {
        FileHandle.standardError.write(Data("preprocess: not yet implemented\n".utf8))
        exit(1)
    }
}
