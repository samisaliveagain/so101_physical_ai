import {
  Application,
  Asset,
  Color,
  Entity,
  FILLMODE_FILL_WINDOW,
  RESOLUTION_AUTO,
  Vec3
} from "../vendor/playcanvas-2.21.0.mjs";

const canvas = document.getElementById("application-canvas");
const status = document.getElementById("status");

function setStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
}

async function loadConfig() {
  const response = await fetch("../config/stage1.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`configuration request failed: HTTP ${response.status}`);
  }
  return response.json();
}

function radians(degrees) {
  return degrees * Math.PI / 180.0;
}

async function main() {
  const config = await loadConfig();
  const app = new Application(canvas, {
    graphicsDeviceOptions: {
      antialias: false,
      alpha: false,
      powerPreference: "high-performance"
    }
  });

  app.setCanvasFillMode(FILLMODE_FILL_WINDOW);
  app.setCanvasResolution(RESOLUTION_AUTO);
  app.scene.ambientLight = new Color(0.35, 0.35, 0.35);
  app.start();

  const camera = new Entity("splat_camera");
  camera.addComponent("camera", {
    clearColor: new Color(0.035, 0.035, 0.045),
    farClip: 1000,
    nearClip: 0.01,
    fov: config.camera.fov_degrees
  });
  app.root.addChild(camera);

  const initial = {
    target: new Vec3(...config.camera.target),
    distance: config.camera.distance,
    yaw: radians(config.camera.yaw_degrees),
    pitch: radians(config.camera.pitch_degrees)
  };
  const view = {
    target: initial.target.clone(),
    distance: initial.distance,
    yaw: initial.yaw,
    pitch: initial.pitch
  };

  const updateCamera = () => {
    const horizontal = Math.cos(view.pitch) * view.distance;
    camera.setPosition(
      view.target.x + Math.sin(view.yaw) * horizontal,
      view.target.y + Math.sin(view.pitch) * view.distance,
      view.target.z + Math.cos(view.yaw) * horizontal
    );
    camera.lookAt(view.target);
  };
  updateCamera();

  let dragging = false;
  let previousX = 0;
  let previousY = 0;
  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    previousX = event.clientX;
    previousY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", (event) => {
    dragging = false;
    canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    view.yaw -= (event.clientX - previousX) * 0.006;
    view.pitch += (event.clientY - previousY) * 0.006;
    view.pitch = Math.max(-1.48, Math.min(1.48, view.pitch));
    previousX = event.clientX;
    previousY = event.clientY;
    updateCamera();
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    view.distance *= Math.exp(event.deltaY * 0.001);
    view.distance = Math.max(0.05, Math.min(100, view.distance));
    updateCamera();
  }, { passive: false });
  window.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() !== "r") return;
    view.target.copy(initial.target);
    view.distance = initial.distance;
    view.yaw = initial.yaw;
    view.pitch = initial.pitch;
    updateCamera();
  });

  const splatAsset = new Asset(
    "Aachen-Mitte",
    "gsplat",
    { url: config.splat }
  );
  app.assets.add(splatAsset);

  const splat = new Entity("scaniverse_workspace");
  splat.setLocalPosition(...config.splat_transform.position);
  splat.setLocalEulerAngles(...config.splat_transform.rotation_degrees);
  splat.setLocalScale(
    config.splat_transform.scale,
    config.splat_transform.scale,
    config.splat_transform.scale
  );

  splatAsset.on("load", () => {
    // Attach only after the PLY resource exists. This avoids a blank unified
    // GSplat component on renderers which do not rebuild an early component.
    // The non-unified path uses the mature per-asset CPU sorter and is more
    // broadly compatible with Gazebo-era OpenGL drivers than the WebGPU-first
    // unified path. Stage 1 has only one splat, so this costs little.
    splat.addComponent("gsplat", { asset: splatAsset, unified: false });
    app.root.addChild(splat);
    setStatus("Loaded 345,369 finite Gaussians");
  });
  splatAsset.on("error", (error) => {
    console.error(error);
    setStatus(`Splat load failed: ${error}`, true);
  });

  setStatus("Loading 89 MB PLY…");
  app.assets.load(splatAsset);

  window.addEventListener("resize", () => app.resizeCanvas());
}

main().catch((error) => {
  console.error(error);
  setStatus(error.message, true);
});
