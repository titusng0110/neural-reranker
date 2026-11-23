const CJK_REGEX = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;

export const MAX_WORD_LEN = 256;

export interface NGramScratchpad {
  maxWordLen: number;
  chars: (string | null)[];
  wordRegex: RegExp;
}

export function createNGramScratchpad(maxWordLen: number = MAX_WORD_LEN): NGramScratchpad {
  return {
    maxWordLen,
    chars: new Array<string | null>(maxWordLen).fill(null),
    wordRegex: /[\p{L}\p{M}\p{N}]+/gu
  };
}

const DEFAULT_SCRATCHPAD: NGramScratchpad = createNGramScratchpad();

function tokenizeWordInto(
  word: string,
  outCollection: Set<string> | string[],
  ctx: NGramScratchpad = DEFAULT_SCRATCHPAD
): void {
  const { maxWordLen, chars } = ctx;
  let len = 0;
  let isCJK = false;

  for (const ch of word) {
    if (len >= maxWordLen) break;
    chars[len++] = ch;
    if (!isCJK && CJK_REGEX.test(ch)) {
      isCJK = true;
    }
  }

  if (len === 0) return;

  const n = isCJK ? 2 : 3;

  if (len < n) {
    if ('add' in outCollection) {
      outCollection.add(word);
    } else {
      outCollection.push(word);
    }
    chars[0] = null;
    return;
  }

  if (n === 2) {
    for (let i = 0; i <= len - 2; i++) {
      const gram = (chars[i] as string) + (chars[i + 1] as string);
      if ('add' in outCollection) {
        outCollection.add(gram);
      } else {
        outCollection.push(gram);
      }
    }
  } else {
    for (let i = 0; i <= len - 3; i++) {
      const gram = (chars[i] as string) + (chars[i + 1] as string) + (chars[i + 2] as string);
      if ('add' in outCollection) {
        outCollection.add(gram);
      } else {
        outCollection.push(gram);
      }
    }
  }

  for (let i = 0; i < len; i++) {
    chars[i] = null;
  }
}

export function tokenizeWord(word: string, ctx: NGramScratchpad = DEFAULT_SCRATCHPAD): string[] {
  const result: string[] = [];
  tokenizeWordInto(word.normalize('NFC').toLowerCase().trim(), result, ctx);
  return result;
}

export function tokenize(text: string, ctx: NGramScratchpad = DEFAULT_SCRATCHPAD): string[] {
  if (!text) return [];

  const clean = /^[\x00-\x7F]*$/.test(text)
    ? text.toLowerCase().trim()
    : text.normalize('NFC').toLowerCase().trim();

  const ngramSet = new Set<string>();
  const regex = ctx.wordRegex;
  regex.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = regex.exec(clean)) !== null) {
    tokenizeWordInto(match[0], ngramSet, ctx);
  }

  return Array.from(ngramSet);
}
