// Camera and face tracking. MediaPipe Face Landmarker runs fully in the
// browser. For every frame it gives two blink scores (0 open .. 1 shut).
//
// The video is NOT mirrored before detection. MediaPipe names the eyes after
// the face it sees, so on an unmirrored frame blendshape "Left" is the
// user's own left eye. Mirroring is only a display option.

const BUNDLE = "/vendor/mediapipe/vision_bundle.mjs";
const WASM_DIR = "/vendor/mediapipe/wasm";
const MODEL = "/models/face_landmarker.task";

export class Tracker {
  // onSample(mpLeft, mpRight, tSeconds, landmarks)
  // onFace(hasFace)
  // onStats({ fps })
  constructor({ video, onSample, onFace, onStats }) {
    this.video = video;
    this.onSample = onSample;
    this.onFace = onFace;
    this.onStats = onStats;
    this.landmarker = null;
    this.stream = null;
    this.running = false;
    this.hasFace = null;
    this._noFace = 0;
    this._lastTs = 0;
    this._lastFrame = 0;
    this._fps = 0;
    this._handle = 0;
    this._usingRvfc = false;
  }

  async load() {
    if (this.landmarker) return;
    const vision = await import(BUNDLE);
    const fileset = await vision.FilesetResolver.forVisionTasks(WASM_DIR);
    const options = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL, delegate },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
    });
    try {
      this.landmarker = await vision.FaceLandmarker.createFromOptions(fileset, options("GPU"));
    } catch {
      this.landmarker = await vision.FaceLandmarker.createFromOptions(fileset, options("CPU"));
    }
  }

  static async listCameras() {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      return all.filter((d) => d.kind === "videoinput");
    } catch {
      return [];
    }
  }

  async startCamera(deviceId = "") {
    this.stopCamera();
    const video = {
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, min: 15 },
    };
    if (deviceId) video.deviceId = { exact: deviceId };
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (err) {
      // A remembered device can disappear. Fall back to the default camera.
      if (deviceId && (err.name === "OverconstrainedError" || err.name === "NotFoundError")) {
        delete video.deviceId;
        stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      } else {
        throw err;
      }
    }
    this.stream = stream;
    // If the camera is unplugged or taken by another app, report "no face".
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      if (this.stream === stream) this.onFace(false);
    });
    this.video.srcObject = stream;
    await this.video.play();
    this.running = true;
    this.hasFace = null;
    this._noFace = 0;
    this._schedule();
  }

  stopCamera() {
    this.running = false;
    if (this._handle) {
      if (this._usingRvfc && this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this._handle);
      else cancelAnimationFrame(this._handle);
      this._handle = 0;
    }
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  _schedule() {
    if (!this.running) return;
    if (this.video.requestVideoFrameCallback) {
      this._usingRvfc = true;
      this._handle = this.video.requestVideoFrameCallback((now) => this._frame(now));
    } else {
      this._usingRvfc = false;
      this._handle = requestAnimationFrame((now) => this._frame(now));
    }
  }

  _frame(now) {
    if (!this.running) return;
    const video = this.video;
    if (this.landmarker && video.readyState >= 2 && video.videoWidth > 0) {
      let ts = performance.now();
      if (ts <= this._lastTs) ts = this._lastTs + 1; // MediaPipe needs strictly rising times
      this._lastTs = ts;
      let result = null;
      try {
        result = this.landmarker.detectForVideo(video, ts);
      } catch {
        result = null;
      }
      this._handleResult(result, ts / 1000);
      this._updateFps(ts);
    }
    this._schedule();
  }

  _handleResult(result, tSeconds) {
    const shapes = result?.faceBlendshapes?.[0]?.categories;
    if (!shapes) {
      this._noFace += 1;
      if (this._noFace >= 3 && this.hasFace !== false) {
        this.hasFace = false;
        this.onFace(false);
      }
      return;
    }
    this._noFace = 0;
    let left = 0;
    let right = 0;
    for (const c of shapes) {
      if (c.categoryName === "eyeBlinkLeft") left = c.score;
      else if (c.categoryName === "eyeBlinkRight") right = c.score;
    }
    if (this.hasFace !== true) {
      this.hasFace = true;
      this.onFace(true);
    }
    this.onSample(left, right, tSeconds, result.faceLandmarks?.[0] || null);
  }

  _updateFps(ts) {
    if (this._lastFrame) {
      const inst = 1000 / Math.max(1, ts - this._lastFrame);
      this._fps = this._fps ? this._fps * 0.9 + inst * 0.1 : inst;
      this.onStats?.({ fps: this._fps });
    }
    this._lastFrame = ts;
  }
}
