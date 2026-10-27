// Capture the README figures from the website report page with local headless Chrome at 2x (light theme).
//   node report/figures/render.mjs [--url <report page>] [name ...]
// The website draws every figure from this repository's bundle (synced into the website), so the README shows the
// same figures in the same style. Run the website first (npm run dev in the website repo) or pass the deployed URL.
// Output: assets/*.{png,jpg}, and assets/scene-explorer.gif from frames of the explorer (assembled by gif.py).
// Loopback DevTools only.
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
// README file name -> website figure class (figure.dl.<class>) and image format
const FIGURES = {
  data: ['data', 'jpeg'], architecture: ['arch', 'jpeg'], recall: ['breakdown', 'png'], flow: ['flow', 'png'],
  confusion: ['confusion', 'png'], examples: ['examples', 'jpeg'], change: ['sensitivity', 'jpeg'], 'next-steps': ['next', 'jpeg'],
};
const args = process.argv.slice(2);
const urlAt = args.indexOf('--url');
const url = urlAt >= 0 ? args[urlAt + 1] : 'http://localhost:4321/projects/disaster-lens/';
const names = args.filter((_, i) => urlAt < 0 || (i !== urlAt && i !== urlAt + 1));
const wanted = names.length ? names : Object.keys(FIGURES);
const profile = await mkdtemp(join(tmpdir(), 'render-figures-'));
const browser = process.env.CHROME_BINARY || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const local = /^https?:\/\/(localhost|127\.0\.0\.1)[:/]/.test(url);
const child = spawn(browser, ['--headless', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--use-mock-keychain', '--password-store=basic', ...(local ? ['--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost'] : []), `--user-data-dir=${profile}`, '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0', 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
const wait = ms => new Promise(r => setTimeout(r, ms));
let socket;
try {
  const endpoint = await new Promise((accept, reject) => {
    const timer = setTimeout(() => reject(new Error('Local Chrome startup timeout')), 15000);
    child.stderr.on('data', b => { const m = b.toString().match(/DevTools listening on (ws:\/\/127\.0\.0\.1:\d+\/\S+)/); if (m) { clearTimeout(timer); accept(m[1]); } });
  });
  socket = new WebSocket(endpoint);
  await new Promise((a, r) => { socket.addEventListener('open', a, {once: true}); socket.addEventListener('error', r, {once: true}); });
  let sequence = 0;
  const requests = new Map(), errors = [];
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.exception?.description || 'exception');
    const pending = requests.get(message.id);
    if (pending) { requests.delete(message.id); message.error ? pending.reject(new Error(JSON.stringify(message.error))) : pending.resolve(message.result); }
  });
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++sequence; requests.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params, ...(sessionId ? {sessionId} : {})}));
  });
  const {targetId} = await call('Target.createTarget', {url: 'about:blank'});
  const {sessionId} = await call('Target.attachToTarget', {targetId, flatten: true});
  const page = (method, params) => call(method, params, sessionId);
  const evaluate = async expression => (await page('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true})).result.value;
  await page('Page.enable'); await page('Runtime.enable');
  await mkdir(join(root, 'assets'), {recursive: true});
  for (const name of wanted) if (name !== 'gif' && !FIGURES[name]) throw new Error(`Unknown figure ${name}`);
  await page('Emulation.setDeviceMetricsOverride', {width: 1280, height: 1000, deviceScaleFactor: 2, mobile: false});
  await page('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: 'light'}]});
  await page('Page.navigate', {url});
  await wait(1500);
  const ready = await evaluate(`(async () => {
    document.documentElement.setAttribute('data-theme', 'light');
    const style = document.createElement('style');
    style.textContent = 'header, .dl-hint, #btt-btn-container, .dl-lightbox, astro-dev-toolbar { display: none !important; } * { transition: none !important; animation: none !important; }';
    document.head.append(style);
    for (const img of document.images) img.loading = 'eager';
    await document.fonts.ready;
    await Promise.all([...document.images].map(i => i.decode().catch(() => null)));
    return document.querySelectorAll('figure.dl').length;
  })()`);
  if (!ready) throw new Error(`No report figures found at ${url}; is the website running?`);
  await wait(500);
  const missing = await evaluate('[...document.images].filter(i => i.getAttribute("src") && !i.naturalWidth).map(i => i.getAttribute("src"))');
  if (missing.length) throw new Error(`Images failed to load: ${missing.join(', ')}`);
  for (const name of wanted.filter(n => n !== 'gif')) {
    const [selector, format] = FIGURES[name];
    const file = join(root, 'assets', `${name}.${format === 'jpeg' ? 'jpg' : 'png'}`);
    const box = await evaluate(`(() => { const r = document.querySelector('figure.dl.${selector}').getBoundingClientRect(); return {x: r.x + scrollX, y: r.y + scrollY, width: Math.ceil(r.width), height: Math.ceil(r.height)}; })()`);
    const pad = 12;  // keep a margin of the page background around the figure
    const clip = {x: box.x - pad, y: box.y - pad, width: box.width + 2 * pad, height: box.height + 2 * pad, scale: 1};
    const {data} = await page('Page.captureScreenshot', {format, quality: format === 'jpeg' ? 88 : undefined, clip, captureBeyondViewport: true});
    await writeFile(file, Buffer.from(data, 'base64'));
    console.log(`${name}: ${clip.width}x${clip.height} css px -> ${file.replace(root + '/', '')}`);
  }
  if (!names.length || names.includes('gif')) {
    // The explorer's steps are pure CSS (data-step on the figure, --split on the stage): before/after sweep,
    // buildings found, predicted damage. The page shows the README's wildfire scene first.
    const frames = await mkdtemp(join(tmpdir(), 'explorer-frames-'));
    const durations = [];
    await evaluate(`(() => {
      const style = document.createElement('style');
      style.textContent = '.explorer .tip, .explorer .loading, .explorer .knob { display: none !important; }';
      document.head.append(style);
      const root = document.querySelector('figure.explorer'); root.classList.remove('compare');
      root.querySelector('.stage').scrollIntoView({block: 'center'});
      return true;
    })()`);
    const state = (step, split) => evaluate(`(() => { const root = document.querySelector('figure.explorer');
      root.dataset.step = '${step}'; root.querySelector('.stage').style.setProperty('--split', '${split}%'); return true; })()`);
    const plan = [[0, 0, 1300], ...[...Array(10)].map((_, i) => [0, Math.round(100 * (i + 1) / 10), 90]), [0, 100, 1300], [1, 100, 1900], [2, 100, 3600]];
    for (const [i, [step, split, ms]] of plan.entries()) {
      await state(step, split); await wait(120);
      const box = await evaluate(`(() => { const r = document.querySelector('figure.explorer .stage').getBoundingClientRect(); return {x: r.x + scrollX, y: r.y + scrollY, width: Math.floor(r.width), height: Math.floor(r.height)}; })()`);
      const {data} = await page('Page.captureScreenshot', {format: 'png', clip: {...box, scale: 1}, captureBeyondViewport: true});
      await writeFile(join(frames, `${String(i).padStart(3, '0')}.png`), Buffer.from(data, 'base64'));
      durations.push(ms);
    }
    await writeFile(join(frames, 'durations.json'), JSON.stringify(durations));
    const python = spawn('uv', ['run', '--offline', '--no-sync', 'python', join(root, 'report/figures/gif.py'), frames, join(root, 'assets/scene-explorer.gif')], {cwd: root, stdio: ['ignore', 'inherit', 'inherit']});
    const code = await new Promise(done => python.on('close', done));
    await rm(frames, {recursive: true, force: true});
    if (code !== 0) throw new Error('GIF assembly failed');
  }
  if (errors.length) throw new Error(`JavaScript errors: ${errors.join(' | ')}`);
  await call('Browser.close');
} finally {
  if (socket) socket.close();
  child.kill();
  await wait(500);
  await rm(profile, {recursive: true, force: true});
}
