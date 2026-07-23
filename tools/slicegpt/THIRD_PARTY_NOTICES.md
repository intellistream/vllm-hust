# SliceGPT third-party notices

`tools/slicegpt/slicegpt/` is a vendored and locally adapted copy of the
SliceGPT implementation from Microsoft:

- Upstream repository: <https://github.com/microsoft/TransformerCompression>
- Upstream source directory: `src/slicegpt`
- Reviewed upstream revision: `6b12cdee6ad51791d7c776b3a046bc408b9e77e9`
- Local adaptations: Qwen2/Qwen2.5 adapters, vLLM checkpoint conversion,
  fail-closed checkpoint validation, and vLLM runtime integration.

The upstream project is distributed under the MIT License.  Its copyright and
license are preserved below and in vendored source-file headers.

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
