/**
 * Internationalization (i18n) for Vision AI Demo
 * Supports English and Japanese
 */

export type Language = 'en' | 'ja';
export type FeatureKey = 'query' | 'caption' | 'detection' | 'point' | 'mask';

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
  actionUpdate: string;
  actionCancel: string;
  actionSave: string;

  // Model Selection
  aiModel: string;
  modelQwen2VL: string;
  modelSmolVLM: string;
  modelMoondream: string;
  modelQwen2VLDesc: string;
  modelSmolVLMDesc: string;
  modelMoondreamDesc: string;
  modelSectionTitle: string;
  modelSelectLabel: string;
  modelQwen2VLTagline: string;
  modelSmolTagline: string;
  modelMoondreamTagline: string;
  modelQwen2VLFeatures: string[];
  modelSmolFeatures: string[];
  modelMoondreamFeatures: string[];
  smolRealtimeTitle: string;
  smolRealtimeSubtitle: string;
  smolPromptLabel: string;
  smolPromptPlaceholder: string;
  smolPromptHint: string;
  moondreamFeatureLabel: string;
  moondreamFeatureNames: Record<FeatureKey, string>;
  moondreamFeatureDescriptions: Record<FeatureKey, string>;
  responseLengthLabel: string;
  responseLengthPreference: string;
  responseLengthShort: string;
  responseLengthMedium: string;
  responseLengthLong: string;
  detectionInstruction: string;
  pointInstruction: string;
  maskInstruction: string;
  queryInstruction: string;

  // Moondream Features
  featureCaption: string;
  featureQuery: string;
  featureDetection: string;
  featurePoint: string;
  featureMask: string;

  // Input
  customQuery: string;
  customQueryPlaceholder: string;
  enterQuery: string;
  sendQuery: string;
  defaultPrompt: string;

  // Captions
  realTimeCaptions: string;
  noCaptions: string;
  latency: string;
  confidence: string;
  captionPanelTitle: string;
  captionSingular: string;
  captionPlural: string;
  captionsStatusConnected: string;
  captionsStatusDisconnected: string;
  captionConfidenceSuffix: string;
  captionEmptyTitle: string;
  captionEmptyMessage: string;

  // Settings
  settingsTitle: string;
  settingsDialogTitle: string;
  settingsDialogDescription: string;
  videoQuality: string;
  videoQualityLow: string;
  videoQualityMedium: string;
  videoQualityHigh: string;
  captureInterval: string;
  captureIntervalDesc: string;
  currentResolution: string;
  close: string;
  responseLengthHint: string;
  captureIntervalHint: string;
  captureIntervalValue: string;
  cameraResolutionLabel: string;
  cameraResolutionHint: string;
  captureIntervalUnit: string;

  // Language
  language: string;
  languageEnglish: string;
  languageJapanese: string;

  // Connection Status
  connected: string;
  disconnected: string;
  connecting: string;
  statusActive: string;
  statusInactive: string;
  connectionSystemStatus: string;
  connectionWebsocket: string;
  connectionBackend: string;
  connectionVideoStream: string;
  serverUrlLabel: string;
  protocolLabel: string;
  protocolValue: string;
  connectionSettingsButton: string;

  // Model Status
  modelLoaded: string;
  modelLoading: string;
  modelNotLoaded: string;

  // Architecture section
  architectureTitle: string;
  architectureFrontendTitle: string;
  architectureFrontendItems: string[];
  architectureBackendTitle: string;
  architectureBackendItems: string[];
  architectureModelsTitle: string;
  architectureModelsItems: string[];

  // Video streaming
  videoInputTitle: string;
  tabCamera: string;
  tabVideo: string;
  buttonStartCamera: string;
  buttonStopCamera: string;
  buttonLoadVideo: string;
  buttonStopVideo: string;
  videoUrlLabel: string;
  videoUrlPlaceholder: string;
  sampleVideo1: string;
  sampleVideo2: string;
  orDividerText: string;
  uploadVideoLabel: string;
  chooseVideoFile: string;
  clearFile: string;
  tipsTitle: string;
  tipsList: string[];
  videoPlaceholder: string;
  errorCameraAccess: string;
  errorVideoNotReady: string;
  errorInvalidVideo: string;
  errorYouTubeCors: string;
  errorVideoRequired: string;
  errorVideoLoad: string;
  errorVideoLoadGeneric: string;

  // Toasts / notifications
  toastStreamStarted: string;
  toastStreamStopped: string;
  toastFeatureSwitched: string; // expects {feature}
  toastModelSwitching: string; // expects {model}
  toastMoondreamUpdated: string;
  toastQuerySent: string;
  toastWsError: string;

  // Additional UI elements
  maskOverlayLabel: string;
  detectionPlaceholder: string;
  processingFrames: string;
  imageLoadError: string;
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
    actionUpdate: 'Update',
    actionCancel: 'Cancel',
    actionSave: 'Save Settings',

    // Model Selection
    aiModel: 'AI Model',
    modelQwen2VL: 'Qwen2-VL',
    modelSmolVLM: 'SmolVLM',
    modelMoondream: 'Moondream',
    modelQwen2VLDesc: 'Multilingual Support',
    modelSmolVLMDesc: 'Fast & Efficient',
    modelMoondreamDesc: 'Advanced Features',
    modelSectionTitle: 'AI Model Configuration',
    modelSelectLabel: 'Select Model',
    modelQwen2VLTagline: 'Native Japanese/Chinese',
    modelSmolTagline: 'Fast & Efficient',
    modelMoondreamTagline: 'Advanced Features',
    modelQwen2VLFeatures: ['Multilingual (EN/JA/ZH/KO)', 'No translation needed', 'Better multilingual accuracy', 'Moderate speed (800ms-1.5s)'],
    modelSmolFeatures: ['Fast inference', 'Low memory usage', 'Custom queries', 'General purpose captions'],
    modelMoondreamFeatures: ['Custom queries', 'Object detection', 'Detailed captions', 'Point detection'],
    smolRealtimeTitle: 'Real-time inference enabled',
    smolRealtimeSubtitle: '· llama.cpp GGUF · <1s response',
    smolPromptLabel: 'Prompt',
    smolPromptPlaceholder: 'Enter a prompt, e.g. "What objects are visible in this scene?"',
    smolPromptHint: 'Prompt will be sent with each video frame',
    moondreamFeatureLabel: 'Moondream Feature',
    moondreamFeatureNames: {
      query: 'Custom Query',
      caption: 'Auto Caption',
      detection: 'Object Detection',
      point: 'Point Detection',
      mask: 'Privacy Mask'
    },
    moondreamFeatureDescriptions: {
      query: 'Ask specific questions about the video content',
      caption: 'Generate descriptive captions automatically',
      detection: 'Identify and locate objects in the video',
      point: 'Detect specific points and coordinates',
      mask: 'Hide/blur detected objects for privacy'
    },
    responseLengthLabel: 'Response Length',
    responseLengthPreference: 'Response length preference',
    responseLengthShort: 'Short',
    responseLengthMedium: 'Medium',
    responseLengthLong: 'Long',
    detectionInstruction: 'Enter one or more comma-separated objects (e.g. "person, car") to highlight them in blue boxes',
    pointInstruction: 'Use a precise label (e.g. "person") to drop a pointer directly on the video stream',
    maskInstruction: 'Enter the objects you would like to blur for privacy (e.g. "face, screen")',
    queryInstruction: 'Ask the model anything about the current scene',

    // Moondream Features
    featureCaption: 'Caption',
    featureQuery: 'Query',
    featureDetection: 'Detect',
    featurePoint: 'Point',
    featureMask: 'Mask',

    // Input
    customQuery: 'Custom Query',
    customQueryPlaceholder: 'Ask about the image...',
    enterQuery: 'Enter your question',
    sendQuery: 'Send',
    defaultPrompt: 'What objects are visible in this scene?',

    // Captions
    realTimeCaptions: 'Real-time Captions',
    noCaptions: 'No captions yet. Start streaming to begin.',
    latency: 'Latency',
    confidence: 'Confidence',
    captionPanelTitle: 'Real-time Response',
    captionSingular: 'caption',
    captionPlural: 'captions',
    captionsStatusConnected: 'Connected to WebSocket',
    captionsStatusDisconnected: 'Disconnected',
    captionConfidenceSuffix: '% confident',
    captionEmptyTitle: 'No captions yet',
    captionEmptyMessage: 'Start video streaming and ensure the WebSocket connection is active to receive real-time captions.',

    // Settings
    settingsTitle: 'Settings',
    settingsDialogTitle: 'Video Processing Settings',
    settingsDialogDescription: 'Configure video capture quality and frame processing rate.',
    videoQuality: 'Video Quality',
    videoQualityLow: 'Low (480p)',
    videoQualityMedium: 'Medium (720p)',
    videoQualityHigh: 'High (1080p)',
    captureInterval: 'Capture Interval',
    captureIntervalDesc: 'seconds between captures',
    currentResolution: 'Current Resolution',
    close: 'Close',
    responseLengthHint: 'Control how much detail the AI provides in its responses.',
    captureIntervalHint: 'How often to capture and process frames. Lower = faster updates but higher processing load.',
    captureIntervalValue: '{ms}ms ({fps} FPS)',
    cameraResolutionLabel: 'Camera Resolution',
    cameraResolutionHint: 'Higher resolution provides more detail but may slow down processing.',
    captureIntervalUnit: 'ms',

    // Language
    language: 'Language',
    languageEnglish: 'English',
    languageJapanese: '日本語',

    // Connection Status
    connected: 'Connected',
    disconnected: 'Disconnected',
    connecting: 'Connecting...',
    statusActive: 'Active',
    statusInactive: 'Inactive',
    connectionSystemStatus: 'System Status',
    connectionWebsocket: 'WebSocket',
    connectionBackend: 'Backend Server',
    connectionVideoStream: 'Video Stream',
    serverUrlLabel: 'Server URL:',
    protocolLabel: 'Protocol:',
    protocolValue: 'WebRTC + WebSocket',
    connectionSettingsButton: 'Settings',

    // Model Status
    modelLoaded: 'Model Loaded',
    modelLoading: 'Loading Model...',
    modelNotLoaded: 'Model Not Loaded',

    // Video streaming
    videoInputTitle: 'Video Input',
    tabCamera: 'Live Camera',
    tabVideo: 'Video File/URL',
    buttonStartCamera: 'Start Camera',
    buttonStopCamera: 'Stop Camera',
    buttonLoadVideo: 'Load Video',
    buttonStopVideo: 'Stop Video',
    videoUrlLabel: 'Video URL',
    videoUrlPlaceholder: 'Paste video URL or try sample videos below',
    sampleVideo1: 'Sample Video 1',
    sampleVideo2: 'Sample Video 2',
    orDividerText: 'OR',
    uploadVideoLabel: 'Upload Video File',
    chooseVideoFile: 'Choose Video File',
    clearFile: 'Clear File',
    tipsTitle: '💡 Tips:',
    tipsList: [
      'Use the sample videos above for quick testing',
      'Upload local files (MP4, WebM, OGG) for best results',
      'Most websites block video embedding due to CORS—upload instead',
      'For YouTube: download the video first, then upload it'
    ],
    videoPlaceholder: 'Video will appear here',
    errorCameraAccess: 'Failed to access camera. Please ensure camera permissions are granted.',
    errorVideoNotReady: 'Video element not ready. Please try again.',
    errorInvalidVideo: 'Please select a valid video file',
    errorYouTubeCors: 'YouTube URLs are not directly supported due to CORS restrictions. Please upload a video file or use a direct video URL (MP4, WebM).',
    errorVideoRequired: 'Please provide a video URL or upload a video file',
    errorVideoLoad: 'Failed to load video. This usually means: 1) Invalid URL, 2) CORS restrictions, or 3) Unsupported format. Try using the sample videos or upload a local file instead.',
    errorVideoLoadGeneric: 'Failed to load video.',

    // Toasts
    toastStreamStarted: 'Video stream started successfully.',
    toastStreamStopped: 'Video stream stopped.',
    toastFeatureSwitched: 'Switched to {feature}.',
    toastModelSwitching: 'Switching to {model} model...',
    toastMoondreamUpdated: 'Moondream settings updated.',
    toastQuerySent: 'Query sent to {model}.',
    toastWsError: 'Unable to update — WebSocket not connected.',

    architectureTitle: 'System Architecture',
    architectureFrontendTitle: 'Frontend (React)',
    architectureFrontendItems: ['WebRTC camera capture', 'YouTube/video file ingestion', 'Real-time WebSocket updates', 'Model configuration controls'],
    architectureBackendTitle: 'Backend (FastAPI)',
    architectureBackendItems: ['Video frame processing', 'WebRTC handshake', 'WebSocket broadcasting', 'Model inference orchestration'],
    architectureModelsTitle: 'AI Models',
    architectureModelsItems: ['SmolVLM (fast, efficient)', 'Moondream (advanced capabilities)', 'Mac M-Chip & NVIDIA GPU support', 'Real-time inference pipeline'],

    // Additional UI elements
    maskOverlayLabel: 'MASKED',
    detectionPlaceholder: "e.g., 'person', 'car', 'book', or leave empty for all objects",
    processingFrames: 'Processing video frames...',
    imageLoadError: 'Error loading image',
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
    actionUpdate: '更新',
    actionCancel: 'キャンセル',
    actionSave: '設定を保存',

    // Model Selection
    aiModel: 'AIモデル',
    modelQwen2VL: 'Qwen2-VL',
    modelSmolVLM: 'SmolVLM',
    modelMoondream: 'Moondream',
    modelQwen2VLDesc: '多言語対応',
    modelSmolVLMDesc: '高速・効率的',
    modelMoondreamDesc: '高度な機能',
    modelSectionTitle: 'AIモデル設定',
    modelSelectLabel: 'モデルを選択',
    modelQwen2VLTagline: 'ネイティブ日本語・中国語',
    modelSmolTagline: '高速・効率的',
    modelMoondreamTagline: '高度な機能',
    modelQwen2VLFeatures: ['多言語対応（英/日/中/韓）', '翻訳不要', '多言語精度向上', '中速（800ms-1.5秒）'],
    modelSmolFeatures: ['高速推論', '低メモリ使用量', 'カスタムクエリ対応', '汎用キャプション'],
    modelMoondreamFeatures: ['カスタムクエリ', '物体検出', '詳細なキャプション', '位置推定'],
    smolRealtimeTitle: 'リアルタイム推論が有効',
    smolRealtimeSubtitle: '· llama.cpp GGUF · 1秒未満の応答',
    smolPromptLabel: 'プロンプト',
    smolPromptPlaceholder: '例：「このシーンに見えるものは？」「人物は何をしていますか？」など',
    smolPromptHint: '各フレーム送信時にプロンプトが使用されます',
    moondreamFeatureLabel: 'Moondream機能',
    moondreamFeatureNames: {
      query: 'カスタムクエリ',
      caption: '自動キャプション',
      detection: '物体検出',
      point: 'ポイント検出',
      mask: 'プライバシーマスク'
    },
    moondreamFeatureDescriptions: {
      query: '動画に関する質問に答えます',
      caption: '動画内容を自動で説明します',
      detection: '動画内の物体を検出して表示します',
      point: '指定した物体の位置をポイントで示します',
      mask: '指定した物体をぼかして非表示にします'
    },
    responseLengthLabel: '応答の長さ',
    responseLengthPreference: '応答の長さの指定',
    responseLengthShort: '短い',
    responseLengthMedium: '普通',
    responseLengthLong: '長い',
    detectionInstruction: 'カンマ区切りで物体名を入力すると（例：「人, 車」）青い枠で強調表示されます',
    pointInstruction: '正確なラベル（例：「人」）を入力すると、動画上にポイントを表示します',
    maskInstruction: 'プライバシー保護のためにぼかしたい物体を入力してください（例：「顔, 画面」）',
    queryInstruction: '現在のシーンについて質問できます',

    // Moondream Features
    featureCaption: 'キャプション',
    featureQuery: 'クエリ',
    featureDetection: '検出',
    featurePoint: 'ポイント',
    featureMask: 'マスク',

    // Input
    customQuery: 'カスタムクエリ',
    customQueryPlaceholder: '画像について質問してください...',
    enterQuery: '質問を入力してください',
    sendQuery: '送信',
    defaultPrompt: 'このシーンに見えるものは何ですか？',

    // Captions
    realTimeCaptions: 'リアルタイム字幕',
    noCaptions: 'まだ字幕がありません。ストリーミングを開始してください。',
    latency: 'レイテンシー',
    confidence: '信頼度',
    captionPanelTitle: 'リアルタイム応答',
    captionSingular: '件の字幕',
    captionPlural: '件の字幕',
    captionsStatusConnected: 'WebSocketに接続中',
    captionsStatusDisconnected: '未接続',
    captionConfidenceSuffix: '% の確度',
    captionEmptyTitle: 'まだ字幕がありません',
    captionEmptyMessage: 'ストリーミングを開始し、WebSocketが接続されていることを確認してください。',

    // Settings
    settingsTitle: '設定',
    settingsDialogTitle: '動画処理の設定',
    settingsDialogDescription: '動画キャプチャの品質と処理頻度を設定します。',
    videoQuality: 'ビデオ品質',
    videoQualityLow: '低 (480p)',
    videoQualityMedium: '中 (720p)',
    videoQualityHigh: '高 (1080p)',
    captureInterval: 'キャプチャ間隔',
    captureIntervalDesc: 'キャプチャ間の秒数',
    currentResolution: '現在の解像度',
    close: '閉じる',
    responseLengthHint: 'AIの応答の詳細度を調整します。',
    captureIntervalHint: 'フレームの処理頻度。値が低いほど更新が速くなりますが、負荷が高くなります。',
    captureIntervalValue: '{ms}ミリ秒（{fps} FPS）',
    cameraResolutionLabel: 'カメラ解像度',
    cameraResolutionHint: '解像度を高くすると詳細になりますが、処理が遅くなる場合があります。',
    captureIntervalUnit: 'ミリ秒',

    // Language
    language: '言語',
    languageEnglish: 'English',
    languageJapanese: '日本語',

    // Connection Status
    connected: '接続済み',
    disconnected: '切断',
    connecting: '接続中...',
    statusActive: '稼働中',
    statusInactive: '停止',
    connectionSystemStatus: 'システム状況',
    connectionWebsocket: 'WebSocket',
    connectionBackend: 'バックエンド',
    connectionVideoStream: '動画ストリーム',
    serverUrlLabel: 'サーバーURL:',
    protocolLabel: 'プロトコル:',
    protocolValue: 'WebRTC + WebSocket',
    connectionSettingsButton: '設定',

    // Model Status
    modelLoaded: 'モデル読み込み済み',
    modelLoading: 'モデル読み込み中...',
    modelNotLoaded: 'モデル未読み込み',

    // Video streaming
    videoInputTitle: '動画入力',
    tabCamera: 'ライブカメラ',
    tabVideo: '動画ファイル・URL',
    buttonStartCamera: 'カメラ開始',
    buttonStopCamera: 'カメラ停止',
    buttonLoadVideo: '動画を読み込む',
    buttonStopVideo: '動画を停止',
    videoUrlLabel: '動画URL',
    videoUrlPlaceholder: '動画URLを貼り付けるか、サンプル動画をお試しください',
    sampleVideo1: 'サンプル動画 1',
    sampleVideo2: 'サンプル動画 2',
    orDividerText: 'または',
    uploadVideoLabel: '動画ファイルをアップロード',
    chooseVideoFile: '動画ファイルを選択',
    clearFile: 'ファイルをクリア',
    tipsTitle: '💡 ヒント：',
    tipsList: [
      '動作確認には上のサンプル動画が便利です',
      'ローカルのMP4/WEBM/OGGファイルをアップロードすると安定します',
      '多くのサイトはCORS制限のため動画埋め込みを許可していません',
      'YouTubeの場合は先に動画をダウンロードしてからアップロードしてください'
    ],
    videoPlaceholder: 'ここに動画が表示されます',
    errorCameraAccess: 'カメラにアクセスできませんでした。権限を確認してください。',
    errorVideoNotReady: '動画要素が準備できていません。再度お試しください。',
    errorInvalidVideo: '有効な動画ファイルを選択してください。',
    errorYouTubeCors: 'YouTubeのURLはCORS制限のため直接は使用できません。動画ファイルをアップロードするか、直接アクセス可能なURLを使用してください。',
    errorVideoRequired: '動画URLを入力するか、動画ファイルをアップロードしてください。',
    errorVideoLoad: '動画を読み込めませんでした。（URLが無効 / CORS制限 / 非対応形式）サンプル動画またはローカルファイルをご利用ください。',
    errorVideoLoadGeneric: '動画の読み込みに失敗しました。',

    // Toasts
    toastStreamStarted: '動画ストリームを開始しました。',
    toastStreamStopped: '動画ストリームを停止しました。',
    toastFeatureSwitched: '{feature}に切り替えました。',
    toastModelSwitching: '{model}モデルに切り替えています...',
    toastMoondreamUpdated: 'Moondreamの設定を更新しました。',
    toastQuerySent: '{model}にクエリを送信しました。',
    toastWsError: '更新できませんでした（WebSocket未接続）。',

    architectureTitle: 'システム構成',
    architectureFrontendTitle: 'フロントエンド（React）',
    architectureFrontendItems: ['WebRTCによるカメラ取得', 'YouTube / 動画ファイルの取り込み', 'リアルタイムWebSocket更新', 'モデル設定コントロール'],
    architectureBackendTitle: 'バックエンド（FastAPI）',
    architectureBackendItems: ['動画フレーム処理', 'WebRTCハンドシェイク', 'WebSocket配信', 'モデル推論の制御'],
    architectureModelsTitle: 'AIモデル',
    architectureModelsItems: ['SmolVLM（高速・効率的）', 'Moondream（高度な機能）', 'Mac Mチップ & NVIDIA GPU対応', 'リアルタイム推論パイプライン'],

    // Additional UI elements
    maskOverlayLabel: 'マスク済み',
    detectionPlaceholder: '例：「人」「車」「本」、または空白ですべての物体を検出',
    processingFrames: '動画フレーム処理中...',
    imageLoadError: '画像の読み込みに失敗しました',
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
  const value = trans[key];
  return (typeof value === 'string' ? value : key) as string;
}
