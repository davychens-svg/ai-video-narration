/**
 * Internationalization (i18n) for Vision AI Demo
 * Supports English and Japanese
 */

export type Language = 'en' | 'ja';

export interface Translations {
  // Header
  title: string;
  subtitle: string;

  // Actions
  start: string;
  stop: string;
  clear: string;
  export: string;
  settings: string;

  // Model Selection
  aiModel: string;
  modelSmolVLM: string;
  modelMoondream: string;
  modelSmolVLMDesc: string;
  modelMoondreamDesc: string;

  // Moondream Features
  featureCaption: string;
  featureQuery: string;
  featureDetection: string;
  featurePoint: string;

  // Input
  customQuery: string;
  customQueryPlaceholder: string;
  enterQuery: string;
  sendQuery: string;

  // Captions
  realTimeCaptions: string;
  noCaptions: string;
  latency: string;
  confidence: string;

  // Settings
  settingsTitle: string;
  videoQuality: string;
  videoQualityLow: string;
  videoQualityMedium: string;
  videoQualityHigh: string;
  captureInterval: string;
  captureIntervalDesc: string;
  currentResolution: string;
  close: string;

  // Language
  language: string;
  languageEnglish: string;
  languageJapanese: string;

  // Connection Status
  connected: string;
  disconnected: string;
  connecting: string;

  // Model Status
  modelLoaded: string;
  modelLoading: string;
  modelNotLoaded: string;
}

export const translations: Record<Language, Translations> = {
  en: {
    // Header
    title: 'Vision AI Demo',
    subtitle: 'Real-time video analysis with Vision-Language Models',

    // Actions
    start: 'Start',
    stop: 'Stop',
    clear: 'Clear',
    export: 'Export',
    settings: 'Settings',

    // Model Selection
    aiModel: 'AI Model',
    modelSmolVLM: 'SmolVLM',
    modelMoondream: 'Moondream',
    modelSmolVLMDesc: 'Fast & Efficient',
    modelMoondreamDesc: 'Advanced Features',

    // Moondream Features
    featureCaption: 'Caption',
    featureQuery: 'Query',
    featureDetection: 'Detect',
    featurePoint: 'Point',

    // Input
    customQuery: 'Custom Query',
    customQueryPlaceholder: 'Ask about the image...',
    enterQuery: 'Enter your question',
    sendQuery: 'Send',

    // Captions
    realTimeCaptions: 'Real-time Captions',
    noCaptions: 'No captions yet. Start streaming to begin.',
    latency: 'Latency',
    confidence: 'Confidence',

    // Settings
    settingsTitle: 'Settings',
    videoQuality: 'Video Quality',
    videoQualityLow: 'Low (480p)',
    videoQualityMedium: 'Medium (720p)',
    videoQualityHigh: 'High (1080p)',
    captureInterval: 'Capture Interval',
    captureIntervalDesc: 'seconds between captures',
    currentResolution: 'Current Resolution',
    close: 'Close',

    // Language
    language: 'Language',
    languageEnglish: 'English',
    languageJapanese: '日本語',

    // Connection Status
    connected: 'Connected',
    disconnected: 'Disconnected',
    connecting: 'Connecting...',

    // Model Status
    modelLoaded: 'Model Loaded',
    modelLoading: 'Loading Model...',
    modelNotLoaded: 'Model Not Loaded',
  },

  ja: {
    // Header
    title: 'ビジョンAIデモ',
    subtitle: 'ビジョン言語モデルによるリアルタイム動画解析',

    // Actions
    start: '開始',
    stop: '停止',
    clear: 'クリア',
    export: 'エクスポート',
    settings: '設定',

    // Model Selection
    aiModel: 'AIモデル',
    modelSmolVLM: 'SmolVLM',
    modelMoondream: 'Moondream',
    modelSmolVLMDesc: '高速・効率的',
    modelMoondreamDesc: '高度な機能',

    // Moondream Features
    featureCaption: 'キャプション',
    featureQuery: 'クエリ',
    featureDetection: '検出',
    featurePoint: 'ポイント',

    // Input
    customQuery: 'カスタムクエリ',
    customQueryPlaceholder: '画像について質問してください...',
    enterQuery: '質問を入力してください',
    sendQuery: '送信',

    // Captions
    realTimeCaptions: 'リアルタイム字幕',
    noCaptions: 'まだ字幕がありません。ストリーミングを開始してください。',
    latency: 'レイテンシー',
    confidence: '信頼度',

    // Settings
    settingsTitle: '設定',
    videoQuality: 'ビデオ品質',
    videoQualityLow: '低 (480p)',
    videoQualityMedium: '中 (720p)',
    videoQualityHigh: '高 (1080p)',
    captureInterval: 'キャプチャ間隔',
    captureIntervalDesc: 'キャプチャ間の秒数',
    currentResolution: '現在の解像度',
    close: '閉じる',

    // Language
    language: '言語',
    languageEnglish: 'English',
    languageJapanese: '日本語',

    // Connection Status
    connected: '接続済み',
    disconnected: '切断',
    connecting: '接続中...',

    // Model Status
    modelLoaded: 'モデル読み込み済み',
    modelLoading: 'モデル読み込み中...',
    modelNotLoaded: 'モデル未読み込み',
  },
};

/**
 * Get translation for current language
 */
export function getTranslations(language: Language): Translations {
  return translations[language] || translations.en;
}

/**
 * Translate a key
 */
export function t(language: Language, key: keyof Translations): string {
  const trans = getTranslations(language);
  return trans[key] || key;
}
