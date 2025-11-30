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
        // Deterministic simplex based on permutation table
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

// Helper for deterministic random based on coordinates
// This ensures that "rand()" in the formula returns the same value for the same block location
const Hash = {
    fract: x => x - Math.floor(x),
    // Standard GLSL hash
    val: (x, z) => {
        return Hash.fract(Math.sin(x * 12.9898 + z * 78.233) * 43758.5453);
    }
}

// FULL CONTEXT WITH ALL VALID FUNCTIONS
const Ctx = {
    // Current Coordinate Context (Updated by Loop)
    _x: 0, _z: 0,

    // CONSTANTS
    pi: Math.PI, 'π': Math.PI,
    e: Math.E,
    phi: 1.618033988749895, 'φ': 1.618033988749895,
    zeta3: 1.202056903159594, 'ζ3': 1.202056903159594,
    catalan: 0.915965594177219, 'K': 0.915965594177219,
    alpha: 2.502907875095892, 'α': 2.502907875095892, feigenbaum: 2.502907875095892,
    delta: 4.669201609102990, 'δ': 4.669201609102990, feigenbaumdelta: 4.669201609102990,
    omega: 0.6889, 'Ω': 0.6889,

    // TRIG
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    csc: x => 1 / Math.sin(x), sec: x => 1 / Math.cos(x), cot: x => 1 / Math.tan(x),
    asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
    acsc: x => Math.asin(1/x), asec: x => Math.acos(1/x), acot: x => Math.atan(1/x),

    // HYPER
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    csch: x => 1/Math.sinh(x), sech: x => 1/Math.cosh(x), coth: x => 1/Math.tanh(x),
    asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh,
    acsch: x => Math.asinh(1/x), asech: x => Math.acosh(1/x), acoth: x => Math.atanh(1/x),

    // ROOT & POWER
    sqrt: Math.sqrt, cbrt: Math.cbrt,
    root: (x, n) => Math.pow(x, 1/n), pow: Math.pow, exp: Math.exp,

    // LOG
    ln: Math.log, lg: Math.log10,

    // NUMBERS
    abs: Math.abs, floor: Math.floor, ceil: Math.ceil, round: Math.round,
    sign: Math.sign, 
    mod: (x, y) => x % y,
    gcd: (x, y) => { x = Math.abs(x); y = Math.abs(y); while(y){var t=y; y=x%y; x=t;} return x; },
    lcm: (x, y) => (!x||!y) ? 0 : Math.abs((x*y)/Ctx.gcd(x,y)),
    modi: (x, m) => (1/x)%m,

    // SPECIAL
    gamma: (z) => { 
        const g=7, p=[0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if(z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * Ctx.gamma(1 - z));
        z -= 1; let x = p[0]; for(let i=1; i<p.length; i++) x += p[i] / (z + i);
        const t = z + g + 0.5; return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
    },
    erf: (x) => {
        let sign = (x >= 0) ? 1 : -1; x = Math.abs(x);
        let a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
        let t = 1.0/(1.0 + p*x);
        return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x));
    },
    beta: (x, y) => Ctx.gamma(x) * Ctx.gamma(y) / Ctx.gamma(x + y),

    // RANDOM (DETERMINISTIC)
    // Uses Ctx._x and Ctx._z to ensure the same coordinate always gets the same random number
    rand: () => Hash.val(Ctx._x, Ctx._z),
    randnormal: (mean, stdev) => {
        // Box-Muller with deterministic seeds
        let u = Hash.val(Ctx._x * 1.5, Ctx._z * 1.5);
        let v = Hash.val(Ctx._x * 2.5, Ctx._z * 2.5);
        if(u<=0) u=0.0001;
        return (Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)) * (stdev||1) + (mean||0);
    },
    randrange: (min, max) => Hash.val(Ctx._x, Ctx._z) * (max - min) + min,

    // NOISE
    perlin: (x, y, z) => Noise.simplex(x, z), 
    simplex: (x, y, z) => Noise.simplex(x, z),
    normal: (x, y, z) => Ctx.randnormal(0, 1),
    blended: (x, y, z) => (Noise.simplex(x, z) + Math.sin(x)*Math.cos(z))/2,
    octaved: (x, z, oct, per) => {
        let total = 0, amp = 1, freq = 1, max = 0;
        for(let i=0; i<(oct||4); i++){
            total += Noise.simplex(x*freq, z*freq)*amp;
            max += amp; amp *= (per||0.5); freq *= 2;
        }
        return total/max;
    },

    // UTIL
    max: Math.max, min: Math.min,
    sigmoid: x => 1 / (1 + Math.exp(-x)),
    clamp: (x, mn, mx) => Math.min(Math.max(x, mn), mx)
};

// ==========================================
// 2. RANDOM FORMULA GENERATOR
// ==========================================

const GenOps = {
    unary: ['sin','cos','tan','abs','floor','ceil','round','sqrt','ln','sigmoid','sign'],
    binary: ['pow','mod','max','min','root'],
    noise: ['perlin','simplex','octaved'],
    ops: ['+','-','*','/'],
    vars: ['x','z'],
    consts: ['pi','e']
};

const NameGen = {
    adj: ['Cosmic','Quantum','Voxel','Hyper','Cyber','Glitch','Floating','Lost','Neon','Dark','Solar','Lunar','Infinite','Fractal','Recursive','Broken','Twisted','Hollow','Solid','Divine'],
    noun: ['Lands','Waves','Mountains','Valley','Spire','Grid','Matrix','Core','Void','Peaks','Dunes','Ocean','Maze','Labyrinth','Citadel','Expanse','Realm','Sector','Zone','Field'],
    get: function() { return this.pick(this.adj) + ' ' + this.pick(this.noun); },
    pick: (arr) => arr[Math.floor(Math.random() * arr.length)]
};

class RandomFormula {
    pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
    
    getScale() {
        // Generates random coordinate scaling like x*0.1 or z*0.05
        const val = Math.random();
        if(val < 0.3) return (Math.random()*0.05 + 0.01).toFixed(4); 
        if(val < 0.7) return (Math.random()*0.2 + 0.05).toFixed(3); 
        return (Math.random()*1.0 + 0.2).toFixed(2); 
    }

    getCoord() {
        const v = this.pick(GenOps.vars);
        const s = this.getScale();
        return `${v} * ${s}`;
    }

    createExpression(depth) {
        if (depth <= 0) {
            const r = Math.random();
            if(r < 0.4) return this.getCoord();
            if(r < 0.5) return this.pick(GenOps.consts);
            if(r < 0.7) return Math.floor(Math.random()*20).toString();
            return `rand() * 10`; 
        }

        const r = Math.random();
        if (r < 0.25) {
            const op = this.pick(GenOps.ops);
            return `(${this.createExpression(depth-1)} ${op} ${this.createExpression(depth-1)})`;
        } else if (r < 0.50) {
            const func = this.pick(GenOps.unary);
            return `${func}(${this.createExpression(depth-1)})`;
        } else if (r < 0.75) {
            const n = this.pick(GenOps.noise);
            if(n === 'octaved') {
                return `octaved(${this.getCoord()}, ${this.getCoord()}, 4, 0.5)`;
            }
            return `${n}(${this.getCoord()}, 0, ${this.getCoord()})`;
        } else {
            const func = this.pick(GenOps.binary);
            return `${func}(${this.createExpression(depth-1)}, ${this.createExpression(depth-1)})`;
        }
    }

    generate(type) {
        let formula = "";
        
        switch(type) {
            case 'HARDCODED':
                formula = `floor(sin(x*0.1) * cos(z*0.1) * 10)`;
                break;
            case 'INTERMEDIATE':
                formula = this.createExpression(3) + " * 8";
                break;
            case 'EXPERT':
                formula = `(${this.createExpression(4)} + ${this.createExpression(2)}) * 15`;
                break;
            case 'UNREAL':
                const base = this.createExpression(3);
                formula = `mod(${base}, 20) * sin(${this.getCoord()}) * 5 + rand()*5`;
                break;
            case 'LONG MATH':
                formula = `${this.createExpression(2)} + ${this.createExpression(2)} - ${this.createExpression(2)}`;
                break;
            default:
                formula = this.createExpression(3) + " * 10";
        }
        
        formula = formula.replace(/\^\^/g, '^'); 
        return formula;
    }
}

const Generator = new RandomFormula();
const Categories = ['HARDCODED', 'INTERMEDIATE', 'EXPERT', 'UNREAL', 'LONG MATH'];

// ==========================================
// 3. THREE.JS SCENE & INFINITE LOGIC
// ==========================================

const container = document.getElementById('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87CEEB);
// Fog hides the chunk boundaries
scene.fog = new THREE.Fog(0x87CEEB, 60, 140);

const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 1000);
const INITIAL_CAM_POS = { x: 50, y: 50, z: 50 };
camera.position.set(INITIAL_CAM_POS.x, INITIAL_CAM_POS.y, INITIAL_CAM_POS.z);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; 
controls.enableZoom = true;
controls.minDistance = 5;
controls.maxDistance = 500;
controls.rotateSpeed = 0.6;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.8;

const ambi = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambi);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(50, 100, 50);
dirLight.castShadow = true;
dirLight.shadow.mapSize.width = 2048;
dirLight.shadow.mapSize.height = 2048;
dirLight.shadow.camera.near = 0.5;
dirLight.shadow.camera.far = 300;
dirLight.shadow.camera.left = -100;
dirLight.shadow.camera.right = 100;
dirLight.shadow.camera.top = 100;
dirLight.shadow.camera.bottom = -100;
scene.add(dirLight);

// INFINITE VOXEL SYSTEM
const GRID = 90; // Render distance (blocks)
const LAYERS = 4;
const TOTAL_INSTANCES = GRID * GRID * LAYERS;

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshStandardMaterial({ 
    color: 0xffffff, 
    roughness: 0.9,
    metalness: 0.1
});

const instMesh = new THREE.InstancedMesh(geometry, material, TOTAL_INSTANCES);
instMesh.castShadow = true;
instMesh.receiveShadow = true;

// CRITICAL FIX: Disable frustum culling so the mesh doesn't disappear when the camera moves far away
instMesh.frustumCulled = false; 

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

// State
let currentFormula = "sin(x*0.1)*cos(z*0.1)*10";
let compiledFunc = null;
let lastCamX = -99999;
let lastCamZ = -99999;

const errorMsg = document.getElementById('error-msg');
function setError(msg) {
    if(msg) {
        errorMsg.textContent = "⚠ " + msg;
        errorMsg.classList.remove('error-hidden');
    } else {
        errorMsg.textContent = "";
        errorMsg.classList.add('error-hidden');
    }
}

function compileFormula(str) {
    try {
        let safeStr = str.replace(/\^/g, '**');
        // We create a function that takes our Context and coordinates
        const f = new Function('C', 'x', 'z', `
            with(C) { 
                return ${safeStr}; 
            }
        `);
        // Test it
        Ctx._x = 0; Ctx._z = 0;
        f(Ctx, 0, 0);
        setError(null);
        return f;
    } catch(e) {
        setError("INVALID SYNTAX OR ARGUMENTS");
        return null;
    }
}

function updateInfiniteTerrain() {
    if(!compiledFunc) return;

    // Grid center based on camera
    const cx = Math.floor(camera.position.x);
    const cz = Math.floor(camera.position.z);

    // Only update if moved 
    if (Math.abs(cx - lastCamX) < 1 && Math.abs(cz - lastCamZ) < 1) return;
    
    lastCamX = cx;
    lastCamZ = cz;
    
    // Move light with camera to maintain shadows
    dirLight.position.set(cx + 50, 100, cz + 50);
    dirLight.target.position.set(cx, 0, cz);
    dirLight.target.updateMatrixWorld();

    let idx = 0;
    const half = GRID / 2;
    
    for(let i = 0; i < GRID; i++) {
        for(let j = 0; j < GRID; j++) {
            // World Coordinates
            const wx = cx - half + i;
            const wz = cz - half + j;
            
            // Set Context Coordinates for deterministic randomness
            Ctx._x = wx;
            Ctx._z = wz;

            let y = 0;
            try { 
                y = compiledFunc(Ctx, wx, wz); 
                if (isNaN(y) || !isFinite(y)) y = 0;
            } catch(e) { y = 0; }

            const surfaceY = Math.floor(y);
            
            // Biome logic
            let biomeType = 'GRASS';
            if (surfaceY < -2) biomeType = 'WATER';
            else if (surfaceY < 2) biomeType = 'SAND';
            else if (surfaceY < 12) biomeType = 'GRASS';
            else if (surfaceY < 30) biomeType = 'STONE';
            else biomeType = 'SNOW';
            
            if (currentFormula.includes('mod') && surfaceY > 5) biomeType = 'NEON'; 
            if (y > 50) biomeType = 'ALIEN';
            if (y < -25) biomeType = 'LAVA';

            for (let d = 0; d < LAYERS; d++) {
                const currentY = surfaceY - d;
                
                dummy.position.set(wx, currentY, wz);
                dummy.updateMatrix();
                instMesh.setMatrixAt(idx, dummy.matrix);

                let c = color;
                if (d === 0) { 
                    if (biomeType === 'WATER') c.copy(colors.water);
                    else if (biomeType === 'SAND') c.copy(colors.sand);
                    else if (biomeType === 'GRASS') c.copy(colors.grass);
                    else if (biomeType === 'STONE') c.copy(colors.stone);
                    else if (biomeType === 'ALIEN') c.copy(colors.alien);
                    else if (biomeType === 'LAVA') c.copy(colors.lava);
                    else if (biomeType === 'NEON') c.copy(colors.neon);
                    else c.copy(colors.snow);
                } 
                else if (d === 1) { 
                    if (biomeType === 'WATER') c.copy(colors.water);
                    else if (biomeType === 'SAND') c.copy(colors.sand);
                    else if (biomeType === 'GRASS') c.copy(colors.dirt);
                    else c.copy(colors.stone);
                } 
                else { 
                    c.copy(colors.stone);
                }

                if (d > 0) c.multiplyScalar(0.9 - (d * 0.1)); 
                instMesh.setColorAt(idx, c);
                idx++;
            }
        }
    }
    instMesh.instanceMatrix.needsUpdate = true;
    instMesh.instanceColor.needsUpdate = true;
}

function updateTerrain(funcStr) {
    currentFormula = funcStr;
    const f = compileFormula(funcStr);
    if(f) {
        compiledFunc = f;
        lastCamX = -99999; // Force redraw
        updateInfiniteTerrain();
    }
}

function animate() {
    requestAnimationFrame(animate);
    updateInfiniteTerrain();
    controls.update();
    renderer.render(scene, camera);
}
animate();

// ==========================================
// 4. UI LOGIC
// ==========================================

const ui = {
    input: document.getElementById('formula-input'),
    tagNoise: document.getElementById('tag-noise'),
    tagGen: document.getElementById('tag-gen'),
    genName: document.getElementById('gen-name'),
    btnGen: document.getElementById('gen-btn'),
    btnCopy: document.getElementById('copy-btn'),
    btnSave: document.getElementById('save-btn'),
    btnHist: document.getElementById('history-btn'),
    btnCloseHist: document.getElementById('close-history'),
    toggleRot: document.getElementById('toggle-rotate'),
    resetCam: document.getElementById('reset-cam'),
    sidebar: document.getElementById('sidebar'),
    sidebarContent: document.getElementById('sidebar-content'),
    toast: document.getElementById('toast'),
    zoomSlider: document.getElementById('zoom-slider'),
    tabHistory: document.getElementById('tab-history'),
    tabSaved: document.getElementById('tab-saved')
};

let historyList = [];
let savedList = [];
let currentTab = 'history'; 

function generateNew() {
    Noise.seed();
    
    // Pick category
    const cat = Generator.pick(Categories);
    const formula = Generator.generate(cat);
    const name = NameGen.get();

    ui.input.value = formula;
    ui.tagNoise.textContent = "RANDOM";
    ui.tagGen.textContent = cat;
    ui.genName.textContent = name;
    
    updateTerrain(formula);
    addToHistory({ formula, type: cat, noise: 'RND', name: name });
}

function addToHistory(data) {
    if(historyList.length > 0 && historyList[0].formula === data.formula) return;
    historyList.unshift(data);
    if(historyList.length > 50) historyList.pop();
    if(currentTab === 'history') renderSidebar();
}

function saveCurrent() {
    const data = {
        formula: ui.input.value,
        type: ui.tagGen.textContent,
        noise: ui.tagNoise.textContent,
        name: ui.genName.textContent || "Custom"
    };
    if(savedList.some(i => i.formula === data.formula)) return;
    savedList.unshift(data);
    ui.toast.textContent = "SAVED!";
    showToast();
    if(currentTab === 'saved') renderSidebar();
}

function renderSidebar() {
    const list = currentTab === 'history' ? historyList : savedList;
    ui.sidebarContent.innerHTML = '';

    if(list.length === 0) {
        ui.sidebarContent.innerHTML = '<div style="color:#666; text-align:center; margin-top:20px;">Empty</div>';
        return;
    }

    list.forEach((item, index) => {
        const el = document.createElement('div');
        el.className = 'history-item';
        
        const content = document.createElement('div');
        content.className = 'history-content';
        content.innerHTML = `
            <div class="h-tags">
                <span class="h-badge n">${item.noise}</span>
                <span class="h-badge t">${item.type}</span>
                <span class="h-name">${item.name}</span>
            </div>
            <span class="h-code">${item.formula}</span>
        `;
        content.onclick = () => {
            ui.input.value = item.formula;
            ui.tagNoise.textContent = item.noise;
            ui.tagGen.textContent = item.type;
            ui.genName.textContent = item.name;
            updateTerrain(item.formula);
        };

        const delBtn = document.createElement('button');
        delBtn.className = 'history-delete';
        delBtn.innerHTML = '×';
        delBtn.onclick = (e) => {
            e.stopPropagation(); 
            list.splice(index, 1);
            renderSidebar();
        };

        el.appendChild(content);
        el.appendChild(delBtn);
        ui.sidebarContent.appendChild(el);
    });
}

ui.input.addEventListener('input', () => {
    ui.tagGen.textContent = "CUSTOM";
    ui.tagNoise.textContent = "USER";
    ui.genName.textContent = "Edited";
    updateTerrain(ui.input.value);
});

let isRotating = true;
ui.toggleRot.addEventListener('click', () => {
    isRotating = !isRotating;
    controls.autoRotate = isRotating;
    ui.toggleRot.classList.toggle('active', isRotating);
});

ui.resetCam.addEventListener('click', () => {
    camera.position.set(INITIAL_CAM_POS.x, INITIAL_CAM_POS.y, INITIAL_CAM_POS.z);
    controls.target.set(camera.position.x-30, 0, camera.position.z-30);
    controls.update();
    ui.zoomSlider.value = 50;
    updateZoom();
});

ui.btnCopy.addEventListener('click', () => {
    const textToCopy = ui.input.value;
    navigator.clipboard.writeText(textToCopy).then(() => {
        ui.toast.textContent = "COPIED";
        showToast();
    }).catch(() => {});
});

function showToast() {
    ui.toast.style.opacity = 1;
    setTimeout(() => ui.toast.style.opacity = 0, 1500);
}

function updateZoom() {
    const pct = parseInt(ui.zoomSlider.value);
    const minD = 5; 
    const maxD = 300;
    const targetDist = maxD - ((maxD - minD) * (pct / 100));
    
    const vec = new THREE.Vector3().subVectors(camera.position, controls.target);
    vec.setLength(targetDist);
    camera.position.copy(controls.target).add(vec);
    controls.update();
}

ui.zoomSlider.addEventListener('input', updateZoom);

ui.btnHist.addEventListener('click', () => {
    ui.sidebar.classList.add('open');
    renderSidebar();
});
ui.btnCloseHist.addEventListener('click', () => ui.sidebar.classList.remove('open'));
ui.btnGen.addEventListener('click', generateNew);
ui.btnSave.addEventListener('click', saveCurrent);

ui.tabHistory.addEventListener('click', () => {
    currentTab = 'history';
    ui.tabHistory.classList.add('active');
    ui.tabSaved.classList.remove('active');
    renderSidebar();
});

ui.tabSaved.addEventListener('click', () => {
    currentTab = 'saved';
    ui.tabSaved.classList.add('active');
    ui.tabHistory.classList.remove('active');
    renderSidebar();
});

window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

generateNew();
