# Speech model license notes

The application remains Apache-2.0. Model and native runtime licenses are
independent of the application license.

| Component | License | Pinned source |
| --- | --- | --- |
| Phonon multilingual 40M weights | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | `gradium/phonon-40m-multilingual@1537ce46c64f42903e0d608b132ccb073dac46d2` |
| Phonon native `ptts` | MIT | `gradium-ai/xn-ptts` release `v0.4.0` |
| Phonon voice catalog | MIT (repository metadata) | `gradium/phonon-voices@0ee17205f254b5abc43b266c8a0f215565c352f7` |
| Granite Speech 5.0 2B | Apache-2.0 | `ibm-granite/granite-speech-5.0-2b` |
| Granite i1-Q4_K_M GGUF | Apache-2.0 (model derivative) | `mradermacher/granite-speech-5.0-2b-i1-GGUF@90042534335e69d091e682a5c6ced21148f79535` |
| Granite Q8_0 audio projector | Apache-2.0 (model derivative) | `mradermacher/granite-speech-5.0-2b-GGUF@e02703d20cf54007c2a5ff6f8dbf923ad26a7519` |
| llama.cpp | MIT | CPU/CUDA/Vulkan server image digests in the workflow and Dockerfiles |

Phonon is for noncommercial use under its model license. Bundling it does not
remove that restriction. Attribution: Gradium, Phonon 40M multilingual; IBM,
Granite Speech 5.0; GGUF conversions by mradermacher. The distributions retain
the original Phonon license texts under `/opt/s2s-models/phonon/licences/`.
Granite weights are installed only after consent; their public Hub repositories
and provenance are linked in the catalog manifest. Upstream model limitations
and quantization accuracy caveats still apply.
