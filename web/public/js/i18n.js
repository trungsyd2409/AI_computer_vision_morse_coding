// UI strings in Vietnamese and English. The same switch also decides the
// language Xeno answers in.

const STRINGS = {
  vi: {
    tagline: "Chat bằng mã Morse, chớp mắt để gõ",
    fpsTitle: "Tốc độ nhận diện khuôn mặt",
    langBtn: "Đổi ngôn ngữ",
    pause: "Tạm dừng", resume: "Tiếp tục", sound: "Âm thanh", settings: "Cài đặt", close: "Đóng",
    you: "Bạn",
    aiFaceLabel: "Khuôn mặt của Xeno Assistant. Mắt mở hoặc đóng theo từng ký hiệu Morse.",
    aiStripLabel: "Mã Morse của AI", userStripLabel: "Mã Morse bạn đang gõ",
    aiLineLabel: "Câu AI đang gõ", userLineLabel: "Câu bạn đang gõ",
    aiPlaceholder: "AI trả lời bằng mã Morse", userPlaceholder: "Chớp mắt để gõ tin nhắn",
    skip: "Bỏ qua", send: "Gửi", clearChat: "Xóa chat",
    chartTitle: "Bảng Morse", chartToggle: "Bảng chữ hoặc số",
    chartLetters: "26 chữ cái", chartDigits: "10 chữ số", chartMatch: "{n} ký tự khớp",
    camEmpty: "Camera chưa bật",
    footKeys: "Bàn phím: Space = chớp (thử), Enter = gửi, Backspace = xóa ký tự cuối",

    ovLead: "Gõ mã Morse bằng cách chớp mắt rồi trò chuyện với Xeno. Xeno cũng trả lời bằng cách chớp mắt.",
    ovHow1: "Chớp ngắn là chấm, nhắm lâu hơn một chút là gạch.",
    ovHow2: "Nghỉ ngắn để sang chữ mới, nghỉ lâu hơn để sang từ mới.",
    ovHow3: "Dừng vài giây sau chữ cuối, tin nhắn tự gửi đi.",
    ovPrivacy: "Video chỉ được xử lý ngay trong trình duyệt và không được tải lên. Chỉ nội dung tin nhắn được gửi tới máy chủ để AI trả lời.",
    ovStartCam: "Bật camera và bắt đầu", ovStartKey: "Chỉ dùng bàn phím (thử)",
    ovModeLive: "AI thật · mô hình {model}", ovModeMock: "AI giả lập · chưa có API key, chỉ trả lời vài câu mẫu",
    ovModeUnknown: "Không kết nối được máy chủ",
    ovLoadingModel: "Đang tải mô hình nhận diện khuôn mặt...",
    ovOpeningCam: "Đang mở camera...",
    ovCamDenied: "Không mở được camera. Hãy cho phép camera trong trình duyệt, hoặc chọn chế độ bàn phím.",
    ovNoModel: "Thiếu tệp nhận diện khuôn mặt. Xem README để biết cách cài. Chế độ bàn phím vẫn dùng được.",
    ovLoadFail: "Không tải được bộ nhận diện: {err}",
    ovNoApi: "Trình duyệt này không hỗ trợ camera. Hãy dùng Chrome hoặc Edge trên máy tính.",

    stReady: "Sẵn sàng", stWaitStart: "Chờ bắt đầu", stCamOn: "Camera bật", stKeyboard: "Chế độ bàn phím",
    stYourTurn: "Đến lượt bạn", stTyping: "Đang gõ", stSending: "Đang gửi", stSent: "Đã gửi",
    stThinking: "Đang nghĩ", stAiTyping: "Đang gõ", stReadingYou: "Đang đọc tin của bạn",
    stReadingAi: "Đang đọc tin của AI", stPaused: "Tạm dừng", stNoFace: "Không thấy khuôn mặt",
    hintNoFace: "Đưa khuôn mặt vào khung hình",
    hintYourTurn: "Chớp mắt để gõ. Dừng {s} giây là tự gửi.",
    sendIn: "{s}s",
    greeting: "XIN CHAO TOI LA XENO",
    who_user: "Bạn", who_ai: "Xeno",
    errReply: "Xeno chưa trả lời được: {msg}",
    errNetwork: "Không kết nối được máy chủ.",
    errRate: "Đã hết lượt trong ngày.",
    tooLong: "Tin nhắn dài tối đa 300 ký tự.",
    muted: "Đã tắt âm", unmuted: "Đã bật âm", paused: "Đã tạm dừng nhập", resumed: "Tiếp tục nhập",
    cleared: "Đã xóa chat",

    setLang: "Ngôn ngữ",
    setLangHint: "Đổi cả giao diện lẫn ngôn ngữ Xeno trả lời. Tiếng Việt được gõ không dấu.",
    setCamera: "Camera", setDevice: "Thiết bị", setMirror: "Lật ngang hình như gương", setSwap: "Đổi mắt trái và mắt phải",
    setThreshold: "Ngưỡng nhắm mắt",
    setThresholdHint: "Giảm nếu chớp mắt không được nhận, tăng nếu bị nhận nhầm.",
    setTyping: "Gõ chữ", setUnit: "Đơn vị thời gian t",
    setUnitHint: "Chấm dưới 1,5 t, gạch từ 1,5 t. Nghỉ 3 t sang chữ mới, 7 t sang từ mới.",
    setSendDelay: "Tự gửi sau khi dừng", setUserBeep: "Phát tiếng bíp khi bạn nhắm mắt",
    setAi: "Xeno Assistant", setPreset: "Tính cách", setPersona: "Mô tả tính cách",
    setPersonaHint: "Sửa nội dung để chuyển sang kiểu Tự viết.",
    presetFunny: "Vui tính", presetCalm: "Điềm đạm", presetTeacher: "Thầy giáo kiên nhẫn", presetCustom: "Tự viết",
    setAiSpeed: "Tốc độ chớp mắt của AI", setMaxChars: "Độ dài tối đa câu trả lời",
    setSound: "Âm thanh", setSoundOn: "Bật tiếng bíp của AI", setVolume: "Âm lượng",
    setChar: "Hình ảnh AI", charDesigned: "Thiết kế sẵn", charCustom: "Ảnh của bạn",
    imgOpen: "Mắt mở", imgClosed: "Mắt nhắm",
    imgHint: "Ảnh chỉ được lưu trong trình duyệt của bạn. Hãy dùng ảnh bạn có quyền sử dụng.",
    imgReset: "Xóa ảnh đã chọn", imgNeedBoth: "Cần chọn cả hai ảnh để dùng ảnh của bạn.",
    imgBad: "Không đọc được ảnh này.",
    resetSettings: "Đặt lại cài đặt",
    unitMs: "{n} ms", seconds: "{n} s",
  },
  en: {
    tagline: "Chat in Morse code, blink to type",
    fpsTitle: "Face tracking speed",
    langBtn: "Change language",
    pause: "Pause", resume: "Resume", sound: "Sound", settings: "Settings", close: "Close",
    you: "You",
    aiFaceLabel: "Xeno Assistant's face. The eyes open and close with every Morse symbol.",
    aiStripLabel: "AI Morse code", userStripLabel: "Morse code you are typing",
    aiLineLabel: "Sentence the AI is typing", userLineLabel: "Sentence you are typing",
    aiPlaceholder: "The AI answers in Morse code", userPlaceholder: "Blink to type a message",
    skip: "Skip", send: "Send", clearChat: "Clear chat",
    chartTitle: "Morse chart", chartToggle: "Letters or digits",
    chartLetters: "26 letters", chartDigits: "10 digits", chartMatch: "{n} matching",
    camEmpty: "Camera is off",
    footKeys: "Keyboard: Space = blink (test), Enter = send, Backspace = delete last character",

    ovLead: "Type Morse code by blinking and chat with Xeno. Xeno answers by blinking too.",
    ovHow1: "A short blink is a dot, a slightly longer one is a dash.",
    ovHow2: "A short pause starts a new letter, a longer pause starts a new word.",
    ovHow3: "Stop for a few seconds after the last letter and the message sends itself.",
    ovPrivacy: "Video is processed inside your browser and is never uploaded. Only the message text goes to the server so the AI can answer.",
    ovStartCam: "Turn on camera and start", ovStartKey: "Keyboard only (test)",
    ovModeLive: "Live AI · model {model}", ovModeMock: "Mock AI · no API key yet, a few sample replies only",
    ovModeUnknown: "Cannot reach the server",
    ovLoadingModel: "Loading the face tracking model...",
    ovOpeningCam: "Opening the camera...",
    ovCamDenied: "Could not open the camera. Allow camera access in your browser, or pick keyboard mode.",
    ovNoModel: "The face tracking file is missing. See the README for setup. Keyboard mode still works.",
    ovLoadFail: "Could not load the face tracker: {err}",
    ovNoApi: "This browser has no camera support. Use Chrome or Edge on a computer.",

    stReady: "Ready", stWaitStart: "Waiting to start", stCamOn: "Camera on", stKeyboard: "Keyboard mode",
    stYourTurn: "Your turn", stTyping: "Typing", stSending: "Sending", stSent: "Sent",
    stThinking: "Thinking", stAiTyping: "Typing", stReadingYou: "Reading your message",
    stReadingAi: "Reading the AI's message", stPaused: "Paused", stNoFace: "No face found",
    hintNoFace: "Move your face into the frame",
    hintYourTurn: "Blink to type. Stop for {s} seconds to send.",
    sendIn: "{s}s",
    greeting: "HELLO I AM XENO",
    who_user: "You", who_ai: "Xeno",
    errReply: "Xeno could not answer: {msg}",
    errNetwork: "Cannot reach the server.",
    errRate: "No messages left for today.",
    tooLong: "Messages can be up to 300 characters.",
    muted: "Sound off", unmuted: "Sound on", paused: "Input paused", resumed: "Input resumed",
    cleared: "Chat cleared",

    setLang: "Language",
    setLangHint: "Changes the interface and the language Xeno answers in. Vietnamese is typed without accents.",
    setCamera: "Camera", setDevice: "Device", setMirror: "Flip the picture like a mirror", setSwap: "Swap left and right eye",
    setThreshold: "Eye closing threshold",
    setThresholdHint: "Lower it if blinks are missed, raise it if you get false blinks.",
    setTyping: "Typing", setUnit: "Time unit t",
    setUnitHint: "A dot is under 1.5 t, a dash is 1.5 t or more. A 3 t pause starts a new letter, 7 t a new word.",
    setSendDelay: "Auto-send after pause", setUserBeep: "Beep while your eyes are closed",
    setAi: "Xeno Assistant", setPreset: "Personality", setPersona: "Personality description",
    setPersonaHint: "Edit the text to switch to the Custom style.",
    presetFunny: "Funny", presetCalm: "Calm", presetTeacher: "Patient teacher", presetCustom: "Custom",
    setAiSpeed: "AI blink speed", setMaxChars: "Maximum reply length",
    setSound: "Sound", setSoundOn: "Play the AI's beep", setVolume: "Volume",
    setChar: "AI picture", charDesigned: "Built-in", charCustom: "Your images",
    imgOpen: "Eyes open", imgClosed: "Eyes closed",
    imgHint: "Images stay in your browser only. Use images you have the right to use.",
    imgReset: "Remove chosen images", imgNeedBoth: "Choose both images to use your own.",
    imgBad: "Could not read this image.",
    resetSettings: "Reset settings",
    unitMs: "{n} ms", seconds: "{n} s",
  },
};

let lang = "vi";
const listeners = new Set();

export function getLang() { return lang; }

export function setLang(next) {
  if (next !== "vi" && next !== "en") return;
  lang = next;
  document.documentElement.lang = next;
  applyI18n(document);
  listeners.forEach((fn) => fn(next));
}

export function onLang(fn) { listeners.add(fn); }

export function t(key, vars) {
  let s = STRINGS[lang][key] ?? STRINGS.vi[key] ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
  return s;
}

export function applyI18n(root) {
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll("[data-i18n-aria]").forEach((el) => { el.setAttribute("aria-label", t(el.dataset.i18nAria)); });
  root.querySelectorAll("[data-i18n-title]").forEach((el) => { el.setAttribute("title", t(el.dataset.i18nTitle)); });
}
