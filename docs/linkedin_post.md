# LinkedIn post — AquaScope

---

A state-of-the-art object detector, pretrained on COCO by people with more H100s than I have brain cells, looked at my 2-inch neon tetra and confidently announced: **"puffin."**

A puffin. The bird. From Iceland.

Eda, my tetra has never even left the tank.

So instead of accepting this — like a normal person and go for a tea break — I built **AquaScope**: an end-to-end on-device CV pipeline that lets me teach a neural net to recognize my fish in the time it takes to finish my tea. Yes, the tea is still there.


How the stack actually works:

→ USB webcam at the tank → V4L2 capture at 720p
→ YOLOv8s in TensorRT FP16, ~7ms/frame on a Jetson Orin Nano
→ ByteTrack on top for stable IDs across occlusions (rock, plant, other tetra, the general existential dread of being a fish)
→ MJPEG dashboard in the browser
→ I sit there with my chaaya and click yes / no on crops like I'm reviewing PRs from a fresher who has never seen a fish
→ Hit **Train Model** → Ultralytics fine-tunes on `dataset/user_recorded/`
→ CUDA path auto-exports a new `.engine` (FP16, or INT8 if you're feeling adventurous)
→ The dashboard hot-swaps the new engine into the live tracker. No restart. No downtime.


The entire pipeline — capture, inference, tracking, labeling UI, fine-tuning, TensorRT compilation, engine hot-swap — runs on a Jetson Orin Nano the size of a matchbox. 25–30 FPS at 720p. 8W of power. Zero cloud. Zero API keys. Zero chance of OpenAI quietly billing me ₹3.3 lakh because a shrimp twitched at 2am.

It also runs on a laptop or a Mac (auto-picks MPS / CUDA / CPU), so you don't need a Jetson and a borderline-unhealthy relationship with your fish to try it. Just one of those is enough. Both is a lifestyle.

Code is AGPL-3.0: github.com/sandheepp/aquascope

If you keep fish, have a Jetson gathering dust, or just want to watch on-device transfer learning actually work — clone it, run `bash scripts/run_local.sh`, kidukku.

No puffins were harmed in the making of this project. The tetra is doing fine. The model has been reformed.

#ComputerVision #EdgeAI #Jetson #TensorRT #YOLO #ByteTrack #MLOps #OnDeviceAI #OpenSource #Kerala #Technopark
