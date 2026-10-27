// Render the README figures from the report bundle with local headless Chrome at 2x.
//   node report/figures/render.mjs [name ...]
// Templates: report/figures/*.html. Numbers: report/bundle/site.json (written to a temporary site.js for the pages).
// Images: report/bundle/images/. Output: assets/*.{png,jpg} and assets/scene-explorer.gif. Loopback DevTools only.
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, readFile, copyFile, rm } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const figures = join(root, 'report/figures');
const FIGURES = {architecture: 'jpeg', data: 'jpeg', recall: 'png', results: 'png', examples: 'jpeg', change: 'jpeg', 'next-steps': 'jpeg'};
const wanted = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(FIGURES);
const site = join(figures, 'site.js');
const profile = await mkdtemp(join(tmpdir(), 'render-figures-'));
const browser = process.env.CHROME_BINARY || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const child = spawn(browser, ['--headless', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--use-mock-keychain', '--password-store=basic', '--host-resolver-rules=MAP * ~NOTFOUND', '--allow-file-access-from-files', `--user-data-dir=${profile}`, '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0', 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
const wait = ms => new Promise(r => setTimeout(r, ms));
let socket;
try {
  await writeFile(site, `window.DATA = ${await readFile(join(root, 'report/bundle/site.json'), 'utf8')};`);
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
  const shots = wanted.map(name => {
    if (!FIGURES[name]) throw new Error(`Unknown figure ${name}`);
    return {page: name, id: 'figure', format: FIGURES[name], file: join(root, 'assets', `${name}.${FIGURES[name] === 'jpeg' ? 'jpg' : 'png'}`)};
  });
  for (const {page: name, id, format, file} of shots) {
    await page('Emulation.setDeviceMetricsOverride', {width: 1300, height: 1000, deviceScaleFactor: 2, mobile: false});
    await page('Page.navigate', {url: pathToFileURL(join(figures, `${name}.html`)).href});
    await wait(700);
    await evaluate('Promise.all([document.fonts.ready, ...[...document.images].map(i => i.decode().catch(() => null))]).then(() => true)');
    await wait(200);
    const missing = await evaluate('[...document.images].filter(i => !i.naturalWidth).map(i => i.getAttribute("src"))');
    if (missing.length) throw new Error(`${name}: images failed to load: ${missing.join(', ')}`);
    const box = await evaluate(`(() => { const r = document.getElementById(${JSON.stringify(id)}).getBoundingClientRect(); return {x: r.x + scrollX, y: r.y + scrollY, width: Math.ceil(r.width), height: Math.ceil(r.height)}; })()`);
    const {data} = await page('Page.captureScreenshot', {format, quality: format === 'jpeg' ? 86 : undefined, clip: {...box, scale: 1}, captureBeyondViewport: true});
    await writeFile(file, Buffer.from(data, 'base64'));
    console.log(`${name}: ${box.width}x${box.height} css px -> ${file.replace(root + '/', '')}`);
  }
  if (errors.length) throw new Error(`JavaScript errors: ${errors.join(' | ')}`);
  await copyFile(join(root, 'report/bundle/media/scene-explorer.gif'), join(root, 'assets/scene-explorer.gif'));
  console.log('scene-explorer.gif -> assets/scene-explorer.gif');
  await call('Browser.close');
} finally {
  if (socket) socket.close();
  child.kill();
  await wait(500);
  await rm(profile, {recursive: true, force: true});
  await rm(site, {force: true});
}
