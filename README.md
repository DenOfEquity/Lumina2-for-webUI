## Lumina2 for webui ##
### only tested with Forge2 ###
I don't think there is anything Forge specific here.


### works for me <sup>TM</sup> on 8GB VRAM (GTX1070) ###

---
## Install ##
Go to the **Extensions** tab, then **Install from URL**, use the URL for this repository.
### needs updated *diffusers* ###

Easiest way to ensure necessary versions are installed is to edit `requirements_versions.txt` in the webUI folder.
```
diffusers>=0.33.0
accelerate>=0.26.0
```

>[!IMPORTANT]
>new *diffusers* has changed the *FlowMatchEulerDiscrete* scheduler; updating will break Flux in Forge.
>Fix:
>* edit `backend\modules\k_prediction.py` line 300
>* `        sigmas = math.exp(self.mu) / (math.exp(self.mu) + (1 / sigmas - 1) ** 1.0)`
>* The current line is `        sigmas = FlowMatchEulerDiscreteScheduler ...`


---
### downloads models on demand - minimum will be ~10GB, if you already have the text encoder (Lumina2 uses the same text encoder as Sana). ###

---
>[!NOTE]
> if **noUnload** is selected then models are kept in memory; otherwise reloaded for each run. The **unload models** button removes them from memory.


---
<details>
<summary>Change log</summary>
2025-04-10: first release. Lumina2 runs quite slow compared to similarly sized models, but seems to have good prompt comprehension and can produce high quality results. Lowering **Shift** and **Max shift** can be helpful. Not sure if loras work.

</details>


---

