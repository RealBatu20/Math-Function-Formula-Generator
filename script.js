import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ==========================================
// 1. MATH ENGINE & CONTEXT
// ==========================================

const Noise = {
    perm: new Uint8Array(512),
    seed: function() {
        const p = new Uint8Array(256);
        for(let i=0; i<256; i++) p[i] = i;
        for(let i=0; i<256; i++) {
            let r = Math.floor(Math.random()*256);
            let t = p[i]; p[i] = p[r]; p[r] = t;
        }
        for(let i=0; i<512; i++) this.perm[i] = p[i & 255];
    },
    grad3: [[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]],
    dot: (g, x, y) => g[0]*x + g[1]*y,
    simplex: function(x, y) {
        const F2 = 0.5*(Math.sqrt(3.0)-1.0);
        const G2 = (3.0-Math.sqrt(3.0))/6.0;
        let n0, n1, n2;
        let s = (x+y)*F2; 
        let i = Math.floor(x+s); let j = Math.floor(y+s);
        let t = (i+j)*G2;
        let X0 = i-t; let Y0 = j-t;
        let x0 = x-X0; let y0 = y-Y0;
        let i1, j1;
        if(x0>y0){i1=1; j1=0;}else{i1=0; j1=1;}
        let x1 = x0 - i1 + G2; let y1 = y0 - j1 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2; let y2 = y0 - 1.0 + 2.0 * G2;
        let ii = i & 255; let jj = j & 255;
        let gi0 = this.perm[ii+this.perm[jj]] % 12;
        let gi1 = this.perm[ii+i1+this.perm[jj+j1]] % 12;
        let gi2 = this.perm[ii+1+this.perm[jj+1]] % 12;
        let t0 = 0.5 - x0*x0 - y0*y0;
        if(t0<0) n0 = 0.0; else {t0 *= t0; n0 = t0 * t0 * this.dot(this.grad3[gi0], x0, y0);}
        let t1 = 0.5 - x1*x1 - y1*y1;
        if(t1<0) n1 = 0.0; else {t1 *= t1; n1 = t1 * t1 * this.dot(this.grad3[gi1], x1, y1);}
        let t2 = 0.5 - x2*x2 - y2*y2;
        if(t2<0) n2 = 0.0; else {t2 *= t2; n2 = t2 * t2 * this.dot(this.grad3[gi2], x2, y2);}
        return 70.0 * (n0 + n1 + n2);
    }
};
Noise.seed();

const Hash = {
    intHash: (x, z) => {
        let h = 0x811c9dc5;
        h ^= (x & 0xFFFFFFFF);
        h = Math.imul(h, 0x01000193);
        h ^= (z & 0xFFFFFFFF);
        h = Math.imul(h, 0x01000193);
        return (h >>> 0) / 4294967296;
    },
    val: (x, z) => {
        const xi = Math.floor(x * 100);
        const zi = Math.floor(z * 100);
        return Hash.intHash(xi, zi);
    }
}

const Ctx = {
    _x: 0, _z: 0,
    pi: Math.PI, 'π': Math.PI, e: Math.E, phi: 1.61803,
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    abs: Math.abs, floor: Math.floor, ceil: Math.ceil, round: Math.round,
    sqrt: Math.sqrt, pow: Math.pow,
    mod: (x, y) => ((x % y) + y) % y,
    max: Math.max, min: Math.min,
    csc: x => 1/Math.sin(x), sec: x => 1/Math.cos(x),
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    ln: Math.log, lg: Math.log10, exp: Math.exp,
    rand: () => Hash.intHash(Ctx._x, Ctx._z),
    randnormal: (mean=0, stdev=1) => {
        let u = Hash.intHash(Ctx._x * 167, Ctx._z * 167);
        let v = Hash.intHash(Ctx._x * 253, Ctx._z * 253);
        if(u<=0) u=0.0001;
        return (Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)) * stdev + mean;
    },
    simplex: (x,y,z) => Noise.simplex(x,z),
    perlin: (x,y,z) => Noise.simplex(x,z),
    octaved: (x, z, oct, per) => {
        let total = 0, amp = 1, freq = 1, max = 0;
        for(let i=0; i<(oct||4); i++){
            total += Noise.simplex(x*freq, z*freq)*amp;
            max += amp; amp *= (per||0.5); freq *= 2;
        }
        return total/max;
    }
};

const Generator = {
    ops: ['+','-','*'],
    funcs: ['sin','cos','abs','floor'],
    noise: ['simplex','octaved'],
    pick: arr => arr[Math.floor(Math.random()*arr.length)],
    genExpr: function(depth) {
        if(depth <= 0) {
            const r = Math.random();
            if(r < 0.6) return (Math.random() < 0.5 ? 'x' : 'z') + '*' + (Math.random()*0.15).toFixed(3);
            return (Math.random()*10).toFixed(1);
        }
        const type = Math.random();
        if(type < 0.3) return `(${this.genExpr(depth-1)} ${this.pick(this.ops)} ${this.genExpr(depth-1)})`;
        if(type < 0.6) return `${this.pick(this.funcs)}(${this.genExpr(depth-1)})`;
        return `${this.pick(this.noise)}(x*0.05, 0, z*0.05) * 10`;
    },
    create: function(type) {
        if(type === 'HARDCODED') return `floor(sin(x*0.1)*cos(z*0.1)*10)`;
        if(type === 'LONG MATH') return `${this.genExpr(2)} + ${this.genExpr(2)}`;
        return `${this.genExpr(3)} * 15`;
    }
};

const Categories = ['HARDCODED', 'INTERMEDIATE', 'EXPERT', 'UNREAL', 'LONG MATH'];

// ==========================================
// 3. THREE.JS SCENE (ISOMETRIC SETUP)
// ==========================================

const container = document.getElementById('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87CEEB);
// NO FOG requested
scene.fog = null; 

// ORTHOGRAPHIC CAMERA (True Isometric)
// View size determines how much of the world is seen (Zoom)
let aspect = container.clientWidth / container.clientHeight;
let viewSize = 40; // Initial zoom level
const camera = new THREE.OrthographicCamera(
    -viewSize * aspect, viewSize * aspect,
    viewSize, -viewSize,
    1, 2000 // Huge Far plane for isometric distance
);

// Isometric Position: Look from a corner
camera.position.set(500, 500, 500); 
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05;
controls.enableZoom = true; // We handle zoom manually to sync with UI, or let Orbit handle it
controls.zoomSpeed = 1.0;
controls.rotateSpeed = 0.5;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.5;

const ambi = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambi);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.1);
dirLight.position.set(200, 400, 200);
dirLight.castShadow = true;
dirLight.shadow.mapSize.width = 4096; // High Res Shadows
dirLight.shadow.mapSize.height = 4096;
const d = 300; 
dirLight.shadow.camera.left = -d;
dirLight.shadow.camera.right = d;
dirLight.shadow.camera.top = d;
dirLight.shadow.camera.bottom = -d;
dirLight.shadow.camera.far = 1000;
scene.add(dirLight);

// ==========================================
// 4. INFINITE VOXEL SYSTEM
// ==========================================

// 200x200 Grid (40,000 blocks per layer)
const GRID = 200; 
const LAYERS = 4;
const TOTAL_INSTANCES = GRID * GRID * LAYERS;

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshStandardMaterial({ 
    color: 0xffffff, 
    roughness: 0.8,
    metalness: 0.2
});

const instMesh = new THREE.InstancedMesh(geometry, material, TOTAL_INSTANCES);
instMesh.castShadow = true;
instMesh.receiveShadow = true;
instMesh.frustumCulled = false; // Always render
scene.add(instMesh);

const dummy = new THREE.Object3D();
const color = new THREE.Color();
const colors = {
    water: new THREE.Color(0x3273a8),
    sand: new THREE.Color(0xdec28a),
    grass: new THREE.Color(0x56a34c),
    dirt: new THREE.Color(0x795548),
    stone: new THREE.Color(0x808080),
    snow: new THREE.Color(0xffffff),
    alien: new THREE.Color(0x9c27b0),
    lava: new THREE.Color(0xff5722),
    neon: new THREE.Color(0x00e5ff)
};

let currentFormula = "";
let compiledFunc = null;
let lastUpdateX = -999999;
let lastUpdateZ = -999999;

function compileFormula(str) {
    try {
        let safeStr = str.replace(/\^/g, '**');
        const f = new Function('C', 'x', 'z', `with(C){return ${safeStr};}`);
        f(Ctx, 0, 0); 
        document.getElementById('error-msg').classList.add('error-hidden');
        return f;
    } catch(e) {
        document.getElementById('error-msg').textContent = "⚠ " + e.message;
        document.getElementById('error-msg').classList.remove('error-hidden');
        return null;
    }
}

function updateTerrain(force = false) {
    if(!compiledFunc) return;

    // Use controls target (Where camera is looking) as the center
    const cx = Math.floor(controls.target.x);
    const cz = Math.floor(controls.target.z);

    if (!force && Math.abs(cx - lastUpdateX) < 2 && Math.abs(cz - lastUpdateZ) < 2) return;
    lastUpdateX = cx;
    lastUpdateZ = cz;
    
    // Light follows center to ensure shadows exist everywhere
    dirLight.position.set(cx + 200, 400, cz + 200);
    dirLight.target.position.set(cx, 0, cz);
    dirLight.target.updateMatrixWorld();

    let idx = 0;
    const offset = Math.floor(GRID / 2);
    
    for(let i = 0; i < GRID; i++) {
        for(let j = 0; j < GRID; j++) {
            const wx = cx - offset + i;
            const wz = cz - offset + j;
            
            Ctx._x = wx; Ctx._z = wz;

            let y = 0;
            try { y = compiledFunc(Ctx, wx, wz); } catch(e) { y = 0; }
            if(!Number.isFinite(y)) y = 0;
            const surfaceY = Math.floor(y);
            
            let biome = 'GRASS';
            if (surfaceY < -2) biome = 'WATER';
            else if (surfaceY < 2) biome = 'SAND';
            else if (surfaceY < 15) biome = 'GRASS';
            else if (surfaceY < 40) biome = 'STONE';
            else biome = 'SNOW';
            
            if(y > 60) biome = 'ALIEN';
            if(y < -30) biome = 'LAVA';

            for (let d = 0; d < LAYERS; d++) {
                dummy.position.set(wx, surfaceY - d, wz);
                dummy.updateMatrix();
                instMesh.setMatrixAt(idx, dummy.matrix);

                let c = color;
                if(d === 0) {
                    if(biome==='WATER') c.copy(colors.water);
                    else if(biome==='SAND') c.copy(colors.sand);
                    else if(biome==='GRASS') c.copy(colors.grass);
                    else if(biome==='STONE') c.copy(colors.stone);
                    else if(biome==='SNOW') c.copy(colors.snow);
                    else if(biome==='ALIEN') c.copy(colors.alien);
                    else if(biome==='LAVA') c.copy(colors.lava);
                } else {
                    c.copy(biome==='GRASS'?colors.dirt:colors.stone);
                }
                if(d>0) c.multiplyScalar(0.85);
                instMesh.setColorAt(idx, c);
                idx++;
            }
        }
    }
    instMesh.instanceMatrix.needsUpdate = true;
    instMesh.instanceColor.needsUpdate = true;
}

// ==========================================
// 5. UI & EVENTS
// ==========================================

const ui = {
    input: document.getElementById('formula-input'),
    btnGen: document.getElementById('gen-btn'),
    zoomSlider: document.getElementById('zoom-slider'),
};

// Update Orthographic Zoom
// Slider Value: 5 (Far) to 100 (Close)
ui.zoomSlider.addEventListener('input', () => {
    const zoomVal = parseInt(ui.zoomSlider.value);
    
    // In Orthographic, lower Zoom number = Farther away (sees more world)
    // But typical Zoom Logic is Higher = Closer.
    // Let's use camera.zoom property directly
    // Slider 5 -> Zoom 0.2 (Far)
    // Slider 100 -> Zoom 3.0 (Close)
    
    // Mapping:
    const newZoom = zoomVal / 20; // 0.25 to 5.0
    camera.zoom = newZoom;
    camera.updateProjectionMatrix();
});

// Sync slider if OrbitControls changes zoom via scroll
controls.addEventListener('change', () => {
    const val = Math.min(100, Math.max(5, camera.zoom * 20));
    ui.zoomSlider.value = val;
    updateTerrain(); // Update terrain as we move
});

function initGen() {
    const type = Generator.pick(Categories);
    const formula = Generator.create(type);
    ui.input.value = formula;
    compiledFunc = compileFormula(formula);
    updateTerrain(true);
    
    document.getElementById('tag-gen').textContent = type;
    document.getElementById('tag-noise').textContent = 'RND';
    document.getElementById('gen-name').textContent = 'Isometric Map';
}

ui.btnGen.addEventListener('click', () => {
    initGen();
    Noise.seed();
});

ui.input.addEventListener('input', () => {
    compiledFunc = compileFormula(ui.input.value);
    updateTerrain(true);
});

document.getElementById('reset-cam').addEventListener('click', () => {
    controls.target.set(0, 0, 0); // Center at world origin
    camera.position.set(500, 500, 500); // Reset Angle
    camera.zoom = 1;
    camera.updateProjectionMatrix();
    controls.update();
    ui.zoomSlider.value = 20;
    updateTerrain(true);
});

document.getElementById('toggle-rotate').addEventListener('click', (e) => {
    controls.autoRotate = !controls.autoRotate;
    e.currentTarget.classList.toggle('active');
});

window.addEventListener('resize', () => {
    aspect = container.clientWidth / container.clientHeight;
    camera.left = -viewSize * aspect;
    camera.right = viewSize * aspect;
    camera.top = viewSize;
    camera.bottom = -viewSize;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

// Sidebar Logic
document.getElementById('history-btn').addEventListener('click', () => document.getElementById('sidebar').classList.add('open'));
document.getElementById('close-history').addEventListener('click', () => document.getElementById('sidebar').classList.remove('open'));
document.getElementById('copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(ui.input.value);
    const t = document.getElementById('toast');
    t.style.opacity = 1;
    setTimeout(()=>t.style.opacity=0, 1500);
});

initGen();
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
