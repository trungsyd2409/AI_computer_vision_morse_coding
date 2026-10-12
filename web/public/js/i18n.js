// UI strings in Vietnamese and English. The same switch also decides the
// language Xeno answers in.

const STRINGS = {
  vi: {
    skipLink: "Chuyển tới cuộc trò chuyện",
    loginLead: "Nhập mật khẩu để vào.",
    loginPassword: "Mật khẩu", loginButton: "Vào",
    loginEmpty: "Hãy nhập mật khẩu.", loginBad: "Sai mật khẩu.",
    loginLocked: "Nhập sai quá nhiều lần. Hãy thử lại sau vài phút.",
    loginNet: "Không kết nối được máy chủ.",
    logout: "Đăng xuất", sessionExpired: "Phiên đăng nhập đã hết hạn. Trang sẽ tải lại.",

    toolsLabel: "Điều khiển",
    fpsTitle: "Tốc độ nhận diện khuôn mặt",
    langBtn: "Đổi ngôn ngữ",
    pause: "Tạm dừng", resume: "Tiếp tục", sound: "Âm thanh", settings: "Cài đặt", close: "Đóng",
    you: "Bạn",
    aiFaceLabel: "Khuôn mặt của Xeno Assistant. Mắt mở hoặc đóng theo từng ký hiệu Morse.",
    tapeLabel: "Băng giấy Morse", tapeSvgLabel: "Mã Morse đang được gõ",
    tapeIdleUser: "Chớp mắt, mã Morse sẽ in ra ở đây.",
    tapeIdleAi: "Xeno sắp gõ.",
    tapeIdleSend: "Xeno đang nghĩ câu trả lời.",
    userPlaceholder: "Chớp mắt để gõ tin nhắn",
    skip: "Bỏ qua", send: "Gửi", clearChat: "Xóa cuộc trò chuyện",
    chartTitle: "Bảng Morse", chartToggle: "Bảng chữ hoặc số",
    chartLetters: "26 chữ cái", chartDigits: "10 chữ số", chartMatch: "{n} ký tự khớp",
    camEmpty: "Camera chưa bật",
    footKeys: "Space để chớp thử, Enter để gửi, Backspace để xóa ký tự cuối.",

    ovLead: "Chớp mắt để gõ mã Morse. Xeno đọc tin của bạn và chớp mắt trả lời.",
    ovHowDot: "Chớp ngắn là một chấm.",
    ovHowDash: "Nhắm mắt lâu hơn một chút là một gạch.",
    ovHowGap: "Mở mắt nghỉ ngắn để sang chữ mới, nghỉ lâu hơn để sang từ mới.",
    ovHowSend: "Dừng vài giây sau chữ cuối, tin nhắn tự gửi đi.",
    ovPrivacy: "Video chỉ được xử lý ngay trong trình duyệt và không được tải lên. Chỉ nội dung tin nhắn được gửi tới máy chủ để AI trả lời.",
    ovStartCam: "Bật camera và bắt đầu", ovStartKey: "Dùng bàn phím để thử",
    ovModeLive: "Đang dùng AI thật, mô hình {model}.", ovModeMock: "Chưa có API key nên Xeno chỉ trả lời vài câu mẫu.",
    ovModeUnknown: "Không kết nối được máy chủ.",
    ovLoadingModel: "Đang tải mô hình nhận diện khuôn mặt...",
    ovOpeningCam: "Đang mở camera...",
    ovCamDenied: "Không mở được camera. Hãy cho phép camera trong trình duyệt, hoặc chọn chế độ bàn phím.",
    ovNoModel: "Thiếu tệp nhận diện khuôn mặt. Xem README để biết cách cài. Chế độ bàn phím vẫn dùng được.",
    ovLoadFail: "Không tải được bộ nhận diện: {err}",
    ovNoApi: "Trình duyệt này không hỗ trợ camera. Hãy dùng Chrome hoặc Edge trên máy tính.",

    stReady: "Sẵn sàng", stWaitStart: "Chờ bắt đầu", stCamOn: "Camera bật", stKeyboard: "Chế độ bàn phím",
    stYourTurn: "Đến lượt bạn", stTyping: "Đang gõ", stSending: "Đang gửi", stSent: "Đã gửi",
    stThinking: "Đang nghĩ", stAiTyping: "Đang gõ", stReadingYou: "Đang đọc tin của bạn",
    stReadingAi: "Đang đọc tin của Xeno", stPaused: "Tạm dừng", stNoFace: "Không thấy khuôn mặt",
    hintNoFace: "Đưa khuôn mặt vào khung hình",
    sendIn: "{s}s",
    greeting: "XIN CHAO TOI LA XENO",
    who_user: "Bạn", who_ai: "Xeno",
    errReply: "Xeno chưa trả lời được: {msg}",
    errNetwork: "Không kết nối được máy chủ.",
    fallbackNote: "Không model AI nào trả lời được, Xeno dùng câu trả lời mặc định.",
    errRate: "Đã hết lượt trong ngày.",
    tooLong: "Tin nhắn dài tối đa 300 ký tự.",
    muted: "Đã tắt âm", unmuted: "Đã bật âm", paused: "Đã tạm dừng nhập", resumed: "Tiếp tục nhập",
    cleared: "Đã xóa cuộc trò chuyện",

    setLang: "Ngôn ngữ",
    setLangHint: "Đổi cả giao diện lẫn ngôn ngữ Xeno trả lời. Tiếng Việt được gõ không dấu.",
    setCamera: "Camera", setDevice: "Thiết bị", setMirror: "Lật ngang hình như gương", setSwap: "Đổi mắt trái và mắt phải",
    setThreshold: "Ngưỡng nhắm mắt",
    setThresholdHint: "Giảm nếu chớp mắt không được nhận, tăng nếu bị nhận nhầm.",
    setTyping: "Gõ chữ", setUnit: "Đơn vị thời gian t",
    setUnitHint: "Chấm dưới 1,5 t, gạch từ 1,5 t. Nghỉ 3 t sang chữ mới, 7 t sang từ mới.",
    setSendDelay: "Tự gửi sau khi dừng",
    setAi: "Xeno Assistant", setPreset: "Tính cách", setPersona: "Mô tả tính cách",
    setPersonaHint: "Sửa nội dung để chuyển sang kiểu Tự viết.",
    presetFunny: "Vui tính", presetCalm: "Điềm đạm", presetTeacher: "Thầy giáo kiên nhẫn", presetCustom: "Tự viết",
    setAiSpeed: "Tốc độ chớp mắt của Xeno",
    setAiSpeedHint: "Tính theo đơn vị t của bạn, tối đa 3 lần. Mỗi đơn vị của Xeno không ngắn hơn 30 ms.",
    setMaxChars: "Độ dài tối đa câu trả lời",
    setSound: "Âm thanh", setSoundOn: "Tiếng bíp của Xeno", setVolume: "Âm lượng",
    setUserSound: "Tiếng của bạn khi gõ",
    setUserSoundHint: "Một tiếng bíp kéo dài khi bạn nhắm mắt, một tiếng tích khi xong chữ, hai nốt khi gửi.",
    setChar: "Hình ảnh của Xeno", charDesigned: "Thiết kế sẵn", charCustom: "Ảnh của bạn",
    imgOpen: "Mắt mở", imgClosed: "Mắt nhắm",
    imgHint: "Ảnh chỉ được lưu trong trình duyệt của bạn. Hãy dùng ảnh bạn có quyền sử dụng.",
    imgReset: "Xóa ảnh đã chọn", imgNeedBoth: "Cần chọn cả hai ảnh để dùng ảnh của bạn.",
    imgBad: "Không đọc được ảnh này.",
    resetSettings: "Đặt lại cài đặt",
    unitMs: "{n} ms", seconds: "{n} s",
  },
  en: {
    skipLink: "Skip to the conversation",
    loginLead: "Enter the password to continue.",
    loginPassword: "Password", loginButton: "Enter",
    loginEmpty: "Enter the password.", loginBad: "Wrong password.",
    loginLocked: "Too many wrong tries. Try again in a few minutes.",
    loginNet: "Cannot reach the server.",
    logout: "Sign out", sessionExpired: "Your session ended. The page will reload.",

    toolsLabel: "Controls",
    fpsTitle: "Face tracking speed",
    langBtn: "Change language",
    pause: "Pause", resume: "Resume", sound: "Sound", settings: "Settings", close: "Close",
    you: "You",
    aiFaceLabel: "Xeno Assistant's face. The eyes open and close with every Morse symbol.",
    tapeLabel: "Morse tape", tapeSvgLabel: "Morse code being typed",
    tapeIdleUser: "Blink and your Morse code prints here.",
    tapeIdleAi: "Xeno is about to type.",
    tapeIdleSend: "Xeno is thinking of a reply.",
    userPlaceholder: "Blink to type a message",
    skip: "Skip", send: "Send", clearChat: "Clear the conversation",
    chartTitle: "Morse chart", chartToggle: "Letters or digits",
    chartLetters: "26 letters", chartDigits: "10 digits", chartMatch: "{n} matching",
    camEmpty: "Camera is off",
    footKeys: "Space blinks for testing, Enter sends, Backspace deletes the last character.",

    ovLead: "Blink to type Morse code. Xeno reads your message and blinks back.",
    ovHowDot: "A short blink is a dot.",
    ovHowDash: "Keeping your eyes closed a little longer is a dash.",
    ovHowGap: "A short pause with open eyes starts a new letter, a longer one starts a new word.",
    ovHowSend: "Stop for a few seconds after the last letter and the message sends itself.",
    ovPrivacy: "Video is processed inside your browser and is never uploaded. Only the message text goes to the server so the AI can answer.",
    ovStartCam: "Turn on camera and start", ovStartKey: "Try with the keyboard",
    ovModeLive: "Using the live AI, model {model}.", ovModeMock: "No API key yet, so Xeno gives a few sample replies only.",
    ovModeUnknown: "Cannot reach the server.",
    ovLoadingModel: "Loading the face tracking model...",
    ovOpeningCam: "Opening the camera...",
    ovCamDenied: "Could not open the camera. Allow camera access in your browser, or pick keyboard mode.",
    ovNoModel: "The face tracking file is missing. See the README for setup. Keyboard mode still works.",
    ovLoadFail: "Could not load the face tracker: {err}",
    ovNoApi: "This browser has no camera support. Use Chrome or Edge on a computer.",

    stReady: "Ready", stWaitStart: "Waiting to start", stCamOn: "Camera on", stKeyboard: "Keyboard mode",
    stYourTurn: "Your turn", stTyping: "Typing", stSending: "Sending", stSent: "Sent",
    stThinking: "Thinking", stAiTyping: "Typing", stReadingYou: "Reading your message",
    stReadingAi: "Reading Xeno's message", stPaused: "Paused", stNoFace: "No face found",
    hintNoFace: "Move your face into the frame",
    sendIn: "{s}s",
    greeting: "HELLO I AM XENO",
    who_user: "You", who_ai: "Xeno",
    errReply: "Xeno could not answer: {msg}",
    errNetwork: "Cannot reach the server.",
    fallbackNote: "No AI model could answer, so Xeno used a default reply.",
    errRate: "No messages left for today.",
    tooLong: "Messages can be up to 300 characters.",
    muted: "Sound off", unmuted: "Sound on", paused: "Input paused", resumed: "Input resumed",
    cleared: "Conversation cleared",

    setLang: "Language",
    setLangHint: "Changes the interface and the language Xeno answers in. Vietnamese is typed without accents.",
    setCamera: "Camera", setDevice: "Device", setMirror: "Flip the picture like a mirror", setSwap: "Swap left and right eye",
    setThreshold: "Eye closing threshold",
    setThresholdHint: "Lower it if blinks are missed, raise it if you get false blinks.",
    setTyping: "Typing", setUnit: "Time unit t",
    setUnitHint: "A dot is under 1.5 t, a dash is 1.5 t or more. A 3 t pause starts a new letter, 7 t a new word.",
    setSendDelay: "Auto-send after pause",
    setAi: "Xeno Assistant", setPreset: "Personality", setPersona: "Personality description",
    setPersonaHint: "Edit the text to switch to the Custom style.",
    presetFunny: "Funny", presetCalm: "Calm", presetTeacher: "Patient teacher", presetCustom: "Custom",
    setAiSpeed: "Xeno's blink speed",
    setAiSpeedHint: "Counted in your time unit t, up to 3 times faster. Xeno's unit never gets shorter than 30 ms.",
    setMaxChars: "Maximum reply length",
    setSound: "Sound", setSoundOn: "Xeno's beep", setVolume: "Volume",
    setUserSound: "Your own typing sounds",
    setUserSoundHint: "A held beep while your eyes are closed, a tick when a letter is done, two notes when a message sends.",
    setChar: "Xeno's picture", charDesigned: "Built-in", charCustom: "Your images",
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
