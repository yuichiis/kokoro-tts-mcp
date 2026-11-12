# kokoro-tts-mcp

[Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) によるローカル音声合成を、OpenAI TTS API（`POST /v1/audio/speech`）互換のツールとして提供する MCP サーバーです。stdio トランスポートで動作します。

- OpenAI 互換の引数（`input` / `model` / `voice` / `response_format` / `speed` / `instructions`）
- 日本語（misaki + UniDic）と英語に対応
- `mp3` / `opus` / `aac` / `flac` / `wav` / `pcm` 出力
- モデル・辞書の自動ダウンロード（初回のみ）

## 必要環境

- [uv](https://docs.astral.sh/uv/getting-started/installation/)（Python 3.12 は uv が用意します）
- `aac` 形式を使う場合のみ ffmpeg

## セットアップ

GitHub から clone し、以下のコマンドでセットアップします。

```pwsh
git clone https://github.com/yuichiis/kokoro-tts-mcp.git
cd kokoro-tts-mcp
uv sync
```

モデルと辞書を事前にダウンロードします（初回のみ）。

```pwsh
uv run kokoro-tts-mcp --download
```

- `kokoro-v1.0.onnx`（約 310 MiB）
- `voices-v1.0.bin`（約 27 MiB）
- `config.json`（v1.1-zh 語彙。日本語音素に必要）
- UniDic 辞書（約 526 MiB。日本語 G2P 用）

> ダウンロードを省略しても、初回のツール呼び出し時に自動で取得されます。UniDic が不要な場合は `--download --models-only` を使ってください。

## MCP クライアント設定

Claude Desktop、Cursor、opencode などの設定に追加します。`--directory` には clone したディレクトリを指定してください。

```json
{
  "mcpServers": {
    "kokoro-tts-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\path\\to\\kokoro-tts-mcp",
        "kokoro-tts-mcp"
      ],
      "env": {
        "KOKORO_DEFAULT_VOICE": "jf_alpha"
      }
    }
  }
}
```

macOS / Linux では `--directory` に `/path/to/kokoro-tts-mcp` を指定します。

## ボイスの変更

### 1. 呼び出しごとに指定

`create_speech` の `voice` 引数に、OpenAI のボイス名または Kokoro のボイス ID を渡します。

```json
{
  "input": "こんにちは。",
  "voice": "jf_alpha"
}
```

### 2. 既定ボイスを変更

環境変数 `KOKORO_DEFAULT_VOICE` を設定します。MCP クライアントの設定ファイルの `env` に書くのが簡単です。

```json
{
  "mcpServers": {
    "kokoro-tts-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "C:\\path\\to\\kokoro-tts-mcp", "kokoro-tts-mcp"],
      "env": {
        "KOKORO_DEFAULT_VOICE": "jf_alpha"
      }
    }
  }
}
```

未設定時の既定値は `alloy`（`af_alloy`）です。`voice` を省略した呼び出しに適用されます。

### 3. 利用可能なボイスを確認

MCP ツール `list_voices` を呼び出すと、ボイス一覧・既定ボイス・OpenAI 名との対応が JSON で返ります。

Kokoro のボイス ID は「言語 + 性別 + 名前」の形式です。

| プレフィックス | 言語 | 例 |
| --- | --- | --- |
| `af` / `am` | アメリカ英語（女性 / 男性） | `af_heart`, `am_adam` |
| `bf` / `bm` | イギリス英語（女性 / 男性） | `bf_emma`, `bm_george` |
| `jf` / `jm` | 日本語（女性 / 男性） | `jf_alpha`, `jm_kumo` |
| `zf` / `zm` | 中国語 | `zf_xiaobei`, `zm_yunjian` |
| `ef` / `em` | スペイン語 | `ef_dora`, `em_alex` |
| `ff` / `fm` | フランス語 | `ff_siwis` |
| `hf` / `hm` | ヒンディー語 | `hf_alpha`, `hm_omega` |
| `if` / `im` | イタリア語 | `if_sara`, `im_nicola` |
| `pf` / `pm` | ポルトガル語 | `pf_dora`, `pm_alex` |

日本語のボイス:

| ボイス ID | 性別 |
| --- | --- |
| `jf_alpha` | 女性 |
| `jf_gongitsune` | 女性 |
| `jf_nezumi` | 女性 |
| `jf_tebukuro` | 女性 |
| `jm_kumo` | 男性 |

OpenAI のボイス名を使う場合は、次のようにマッピングされます。

| OpenAI | Kokoro |
| --- | --- |
| `alloy` | `af_alloy` |
| `echo` | `am_echo` |
| `fable` | `bm_fable` |
| `onyx` | `am_onyx` |
| `nova` | `af_nova` |
| `shimmer` | `af_sky` |

> 日本語のテキストに英語ボイス（`alloy` など）を指定すると、日本語の音素を英語話者の声質で読み上げます。日本語は `jf_*` / `jm_*` の使用を推奨します。

## ツール

### `create_speech`

OpenAI の `POST /v1/audio/speech` と同じ引数を受け付け、音声を MCP の audio content（base64）で返します。

| 引数 | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `input` | string | （必須） | 合成するテキスト。最大 4096 文字 |
| `model` | `tts-1` / `tts-1-hd` / `gpt-4o-mini-tts` | `tts-1` | 互換用。値は無視されます |
| `voice` | string | `alloy` | OpenAI 名または Kokoro のボイス ID |
| `response_format` | `mp3` / `opus` / `aac` / `flac` / `wav` / `pcm` | `mp3` | `pcm` は 24kHz 16bit LE モノラル |
| `speed` | number | `1.0` | 0.25〜4.0 |
| `instructions` | string | `null` | 互換用。Kokoro では未対応 |
| `language` | `auto` / `ja` / `en` | `auto` | `auto` は日本語ボイス/日本語文字なら `ja`、それ以外は `en` |

### `list_voices`

利用可能な Kokoro ボイス、既定ボイス、OpenAI 名のマッピング、対応フォーマットを JSON で返します。

## モデルのダウンロード

```pwsh
uv run kokoro-tts-mcp --download              # モデル 3 点 + UniDic
uv run kokoro-tts-mcp --download --models-only # UniDic を除く
uv run kokoro-tts-mcp --download --force       # 既存ファイルを取り直す
uv run kokoro-tts-mcp --download --models-dir D:\models
```

保存先は `--models-dir` / `KOKORO_MODEL_DIR`、未指定時はユーザーキャッシュです。

| OS | 既定の保存先 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\kokoro-tts-mcp\models` |
| macOS / Linux | `~/.cache/kokoro-tts-mcp/models` |

リポジトリ直下にモデルがある場合は、そちらが優先して使われます。進捗は stderr に出力されるため、MCP の stdio 通信を妨げません。

## 環境変数

| 変数 | 説明 |
| --- | --- |
| `KOKORO_DEFAULT_VOICE` | `voice` 省略時の既定ボイス（既定: `alloy`） |
| `KOKORO_MODEL` | `kokoro-v1.0.onnx` のパス |
| `KOKORO_VOICES` | `voices-v1.0.bin` のパス |
| `KOKORO_CONFIG` | `config.json` のパス |
| `KOKORO_MODEL_DIR` | モデルを置く/探すディレクトリ |
| `KOKORO_AUTO_DOWNLOAD` | `0` で自動ダウンロードを無効化（既定: `1`） |

ファイルは個別指定 → `KOKORO_MODEL_DIR` → プロジェクトルート → カレントディレクトリ → ユーザーキャッシュ、の順で探します。

## 補足

- `aac` 形式のみ ffmpeg が必要です（PATH 上にある場合）。
- 長いテキストは文単位で分割し、無音を挟んで結合します。
- モデルは初回のツール呼び出し時に遅延ロードされ、不足ファイルはそのタイミングでダウンロードされます。
- ダウンロードは HTTPS（certifi の CA バンドル）で行い、サイズ検証と最大 3 回のリトライ付きです。
- Windows の既定コードページ対策として、サーバーは stdin/stdout を UTF-8 に設定します。

## 構成

```
kokoro-tts-mcp/
├── src/kokoro_tts_mcp/
│   ├── __init__.py
│   └── server.py
├── pyproject.toml
├── uv.lock
└── README.md
```
