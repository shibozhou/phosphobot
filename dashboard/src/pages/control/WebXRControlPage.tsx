import {WebXRSimulator} from "./WebXRSimulator"; 

export function WebXRControlPage() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">WebXR Control</h1>
      <p className="mb-4">
        Use the controls below to simulate a VR controller.
      </p>
      <WebXRSimulator />
    </div>
  );
}