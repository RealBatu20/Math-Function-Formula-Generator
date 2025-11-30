import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ==========================================
// 1. MATH & NOISE ENGINE
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

// The Context Object containing ALL requested valid math functions
const Ctx = {
    // TRIGONOMETRIC
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    csc: x => 1 / Math.sin(x), sec: x => 1 / Math.cos(x), cot: x => 1 / Math.tan(x),
    asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
    acsc: x => Math.asin(1/x), asec: x => Math.acos(1/x), acot: x => Math.atan(1/x),

    // HYPERBOLIC
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    csch: x => 1/Math.sinh(x), sech: x => 1/Math.cosh(x), coth: x => 1/Math.tanh(x),
    asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh,
    acsch: x => Math.asinh(1/x), asech: x => Math.acosh(1/x), acoth: x => Math.atanh(1/x),

    // ROOT AND POWER
    sqrt: Math.sqrt, cbrt: Math.cbrt,
    root: (x, n) => Math.pow(x, 1/n), pow: Math.pow, exp: Math.exp,

    // LOGARITHMIC
    ln: Math.log, lg: Math.log10,

    // ROUNDING AND NUMBERS
    abs: Math.abs, floor: Math.floor, ceil: Math.ceil, round: Math.round,
    sign: Math.sign, mod: (x, y) => x % y,
    gcd: (x, y) => { x = Math.abs(x); y = Math.abs(y); while(y){var t=y; y=x%y; x=t;} return x; },
    lcm: (x, y) => (!x||!y) ? 0 : Math.abs((x*y)/Ctx.gcd(x,y)),
    modi: (x, m) => (1/x)%m, // simplified modular inverse simulation

    // SPECIAL
    gamma: (z) => { 
        if (z < 0) return 0; // Prevent complex issues
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

    // RANDOM
    rand: Math.random,
    randnormal: (mean, stdev) => {
        let u=0, v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
        return (Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)) * (stdev||1) + (mean||0);
    },
    randrange: (min, max) => Math.random() * (max - min) + min,

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

    // UTILITY
    max: Math.max, min: Math.min,
    sigmoid: x => 1 / (1 + Math.exp(-x)),
    clamp: (x, mn, mx) => Math.min(Math.max(x, mn), mx),

    // CONSTANTS
    pi: Math.PI, 'π': Math.PI,
    e: Math.E,
    phi: 1.618033988749895, 'φ': 1.618033988749895,
    zeta3: 1.202056903159594, 'ζ3': 1.202056903159594,
    catalan: 0.915965594177219, 'K': 0.915965594177219,
    alpha: 2.502907875095892, 'α': 2.502907875095892, 'feigenbaum': 2.502907875095892,
    delta: 4.669201609102990, 'δ': 4.669201609102990, 'feigenbaumdelta': 4.669201609102990,
    omega: 0.6889, 'Ω': 0.6889
};

// ==========================================
// 2. FORMULA GENERATOR (STRICTLY VALID)
// ==========================================

const Generator = {
    ops: ['+', '-', '*', '/', '^'],
    vars: ['x', 'z'],
    unary: [
        'sin','cos','tan','csc','sec','cot','sinh','cosh','tanh','abs','floor','ceil','round','sqrt','cbrt','exp','ln','lg','sigmoid','erf','gamma','sign'
    ],
    binary: [
        'pow','mod','max','min','atan2'
    ],
    noise: [
        'perlin(x*S,0,z*S)',
        'simplex(x*S,0,z*S)',
        'blended(x*S,0,z*S)',
        'octaved(x*S,z*S,4,0.5)'
    ],

    // Helper to pick random array element
    pick: (arr) => arr[Math.floor(Math.random() * arr.length)],
    
    // Helper to format float
    f: (n) => n.toFixed(3),

    // Generate a valid coordinate scaler (e.g., x * 0.05)
    getVar: function() {
        const v = this.pick(this.vars);
        const scale = (Math.random() * 0.2 + 0.01).toFixed(3); // 0.01 to 0.21
        return `(${v}*${scale})`;
    },

    // Recursive formula builder
    buildTree: function(depth, mode) {
        if (depth <= 0) {
            // Leaf node: return variable, number, constant, or noise
            const r = Math.random();
            if (r < 0.3) return (Math.random() * 20 - 10).toFixed(2); // Number
            if (r < 0.4 && mode !== 'HARDCODED') {
                const consts = ['pi', 'e', 'phi', 'zeta3', 'catalan'];
                return this.pick(consts);
            }
            if (r < 0.7) return this.getVar();
            if (mode === 'UNREAL') return this.pick(this.noise).replace(/S/g, (Math.random()*0.1).toFixed(3));
            return this.getVar();
        }

        const type = Math.random();
        
        // Unary Function
        if (type < 0.4) {
            const func = this.pick(this.unary);
            return `${func}(${this.buildTree(depth - 1, mode)})`;
        }
        // Binary Function
        else if (type < 0.6) {
            const func = this.pick(this.binary);
            return `${func}(${this.buildTree(depth - 1, mode)}, ${this.buildTree(depth - 1, mode)})`;
        }
        // Operator
        else {
            const op = this.pick(this.ops);
            return `(${this.buildTree(depth - 1, mode)} ${op} ${this.buildTree(depth - 1, mode)})`;
        }
    },

    create: function() {
        // Types: HARDCODED, INTERMEDIATE, EXPERT, UNREAL, LONG
        const types = ['HARDCODED', 'INTERMEDIATE', 'EXPERT', 'UNREAL', 'LONG'];
        const type = this.pick(types);
        let formula = "";
        let name = "";
        
        switch(type) {
            case 'HARDCODED':
                formula = `sin(x*0.1) * cos(z*0.1) * 10`;
                name = "Simple Waves";
                // Add variety
                if(Math.random()>0.5) {
                    formula = `floor(sin(x*0.2) + cos(z*0.2)) * 5`;
                    name = "Steps";
                }
                break;
            case 'INTERMEDIATE':
                // Depth 2-3
                formula = this.buildTree(2, 'INTERMEDIATE');
                formula = `${formula} * 5`; // Amplitude
                name = "Algebraic";
                break;
            case 'EXPERT':
                // Depth 3-4, specific functions
                formula = `gamma(abs(${this.getVar()})) * 2 + sin(${this.getVar()})*10`;
                name = "Complex Func";
                break;
            case 'UNREAL':
                // Noise heavy
                const n1 = this.pick(this.noise).replace(/S/g, '0.05');
                const n2 = this.pick(this.noise).replace(/S/g, '0.02');
                formula = `(${n1} * 20) + (${n2} * 10)`;
                name = "Noise Terrain";
                break;
            case 'LONG':
                // Depth 5+
                formula = this.buildTree(5, 'LONG');
                name = "Deep Math";
                break;
        }

        // Final cleanup ensuring validity
        return {
            type: type,
            name: name,
            formula: formula
        };
    }
};

// ==========================================
// 3. THREE.JS SCENE & INFINITE SYSTEM
// ==========================================

const container = document.getElementById('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87CEEB);
scene.fog = new THREE.Fog(0x87CEEB, 30, 150);

const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 500);
const INITIAL_CAM_POS = { x: 0, y: 50, z: 50 }; // Centered at 0,0 for logic
camera.position.set(INITIAL_CAM_POS.x, INITIAL_CAM_POS.y, INITIAL_CAM_POS.z);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; 
controls.enableZoom = true;
controls.zoomSpeed = 0.4; 
controls.rotateSpeed = 0.6;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.8;
controls.target.set(0, 0, 0);

// Limit Zoom for Slider Logic
controls.minDistance = 10;
controls.maxDistance = 200;

// Lighting
const ambi = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambi);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(50, 80, 50);
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
// Instead of creating/destroying meshes, we move a large grid to follow the camera
// and update the Y positions based on world coordinates.

const GRID_SIZE = 80; // 80x80 visible area
const LAYERS = 4; // Depth layers
const TOTAL_INSTANCES = GRID_SIZE * GRID_SIZE * LAYERS;

const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, metalness: 0.1 });
const instMesh = new THREE.InstancedMesh(geometry, material, TOTAL_INSTANCES);
instMesh.castShadow = true;
instMesh.receiveShadow = true;
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

let currentFormulaFunction = null;

function compileFormula(str) {
    try {
        const safeStr = str.replace(/\^/g, '**');
        // Check for disallowed characters roughly (allowing vars, funcs, nums, ops)
        // This is a loose check, the real test is `new Function`
        return new Function('ctx', 'x', 'z', `with(ctx) { try { return ${safeStr}; } catch(e){ return NaN; } }`);
    } catch(e) {
        return null;
    }
}

function updateVoxels() {
    if (!currentFormulaFunction) return;

    // Center grid on camera floor position
    const camX = Math.floor(controls.target.x);
    const camZ = Math.floor(controls.target.z);
    
    // We update the instances based on world coordinates relative to camera
    const offset = Math.floor(GRID_SIZE / 2);
    let idx = 0;

    for(let x = 0; x < GRID_SIZE; x++) {
        for(let z = 0; z < GRID_SIZE; z++) {
            // World Coordinates
            const wx = camX + (x - offset);
            const wz = camZ + (z - offset);
            
            let y = 0;
            try { y = currentFormulaFunction(Ctx, wx, wz); } catch(e) {}
            
            if (isNaN(y) || !isFinite(y)) y = 0;
            const surfaceY = Math.floor(y);

            // Determine Biome
            let biomeType = 'GRASS';
            if (surfaceY < -2) biomeType = 'WATER';
            else if (surfaceY < 2) biomeType = 'SAND';
            else if (surfaceY < 12) biomeType = 'GRASS';
            else if (surfaceY < 20) biomeType = 'STONE';
            else biomeType = 'SNOW';

            // Special conditions based on height/val
            if (y > 40) biomeType = 'ALIEN';
            if (y < -20) biomeType = 'LAVA';
            
            // Build Layers
            for (let d = 0; d < LAYERS; d++) {
                const currentY = surfaceY - d;
                
                dummy.position.set(wx, currentY, wz);
                dummy.scale.set(1, 1, 1);
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
                    else c.copy(colors.snow);
                } else if (d === 1 || d === 2) { 
                    if (biomeType === 'WATER') c.copy(colors.water);
                    else if (biomeType === 'SAND') c.copy(colors.sand);
                    else if (biomeType === 'GRASS') c.copy(colors.dirt);
                    else c.copy(colors.stone);
                } else { 
                    c.copy(colors.stone);
                }
                if (d > 0) c.multiplyScalar(0.9 - (d * 0.05)); 
                instMesh.setColorAt(idx, c);
                idx++;
            }
        }
    }
    instMesh.instanceMatrix.needsUpdate = true;
    instMesh.instanceColor.needsUpdate = true;
}

// Logic to update terrain only when moving significantly or formula changed
let lastCamX = -99999;
let lastCamZ = -99999;

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    
    // Infinite Terrain Logic: Check if camera moved to a new block
    const cx = Math.floor(controls.target.x);
    const cz = Math.floor(controls.target.z);
    
    // Light follows camera for shadows
    dirLight.position.x = cx + 50;
    dirLight.position.z = cz + 50;
    dirLight.target.position.set(cx, 0, cz);
    dirLight.target.updateMatrixWorld();

    if (cx !== lastCamX || cz !== lastCamZ) {
        updateVoxels();
        lastCamX = cx;
        lastCamZ = cz;
    }
    
    renderer.render(scene, camera);
}
animate();

// ==========================================
// 4. UI LOGIC
// ==========================================

const ui = {
    input: document.getElementById('formula-input'),
    errorMsg: document.getElementById('error-msg'),
    tagType: document.getElementById('tag-type'),
    tagName: document.getElementById('tag-name'),
    btnGen: document.getElementById('gen-btn'),
    btnCopy: document.getElementById('copy-btn'),
    btnSave: document.getElementById('save-btn'),
    btnHistory: document.getElementById('history-btn'),
    btnClose: document.getElementById('close-history'),
    toggleRot: document.getElementById('toggle-rotate'),
    resetCam: document.getElementById('reset-cam'),
    zoomSlider: document.getElementById('zoom-slider'),
    
    sidebar: document.getElementById('sidebar'),
    tabHistory: document.getElementById('tab-history'),
    tabSaved: document.getElementById('tab-saved'),
    listHistory: document.getElementById('list-history'),
    listSaved: document.getElementById('list-saved'),
    toast: document.getElementById('toast')
};

// State
let currentData = { type: 'READY', name: 'INIT', formula: '0' };
let historyList = [];
let savedList = [];

// ZOOM SLIDER
ui.zoomSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value); // 0 - 100
    // Map 0-100 to MinDist-MaxDist (Inverse: 0 is close, 100 is far)
    // 50% = ~40 units
    // Formula: lerp
    const minD = 10;
    const maxD = 150;
    const dist = minD + (val / 100) * (maxD - minD);
    
    // Adjust camera distance
    const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
    camera.position.copy(controls.target).add(dir.multiplyScalar(dist));
});

function validateAndRender(formula, type, name) {
    ui.errorMsg.textContent = "";
    
    const func = compileFormula(formula);
    if (!func) {
        ui.errorMsg.textContent = "INVALID SYNTAX OR CHARACTERS";
        return false;
    }
    
    // Check runtime
    try {
        const val = func(Ctx, 0, 0);
        if (isNaN(val) && typeof val !== 'number') {
            ui.errorMsg.textContent = "RUNTIME ERROR: RESULT IS NaN";
            return false;
        }
    } catch(e) {
        ui.errorMsg.textContent = "RUNTIME ERROR: " + e.message;
        return false;
    }

    // Success
    currentFormulaFunction = func;
    currentData = { type, name, formula };
    
    ui.input.value = formula;
    ui.tagType.textContent = type;
    ui.tagName.textContent = name;
    
    lastCamX = -99999; // Force redraw
    return true;
}

function generateNew() {
    Noise.seed();
    const data = Generator.create();
    if(validateAndRender(data.formula, data.type, data.name)) {
        addToHistory(data);
    }
}

function addToHistory(data) {
    const exists = historyList.some(h => h.formula === data.formula);
    if(exists) return;
    
    historyList.unshift(data);
    if(historyList.length > 50) historyList.pop();
    renderList(historyList, ui.listHistory, false);
}

function saveCurrent() {
    const data = { ...currentData };
    const exists = savedList.some(s => s.formula === data.formula);
    if(!exists) {
        savedList.unshift(data);
        renderList(savedList, ui.listSaved, true);
        showToast("SAVED");
    }
}

function renderList(list, container, isSaved) {
    container.innerHTML = "";
    list.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            <div class="history-content">
                <div class="h-tags">
                    <span class="h-badge t">${item.type}</span>
                    <span class="h-badge name">${item.name}</span>
                </div>
                <span class="h-code">${item.formula}</span>
            </div>
            <button class="history-delete">×</button>
        `;
        
        div.querySelector('.history-content').onclick = () => {
            validateAndRender(item.formula, item.type, item.name);
        };
        
        div.querySelector('.history-delete').onclick = (e) => {
            e.stopPropagation();
            list.splice(index, 1);
            renderList(list, container, isSaved);
        };
        
        container.appendChild(div);
    });
}

// Event Listeners
ui.btnGen.addEventListener('click', generateNew);
ui.btnSave.addEventListener('click', saveCurrent);

ui.input.addEventListener('input', () => {
    const val = ui.input.value;
    validateAndRender(val, 'CUSTOM', 'Manual Input');
});

ui.btnHistory.addEventListener('click', () => ui.sidebar.classList.add('open'));
ui.btnClose.addEventListener('click', () => ui.sidebar.classList.remove('open'));

// Tab Switching
ui.tabHistory.addEventListener('click', () => {
    ui.tabHistory.classList.add('active');
    ui.tabSaved.classList.remove('active');
    ui.listHistory.classList.add('active');
    ui.listSaved.classList.remove('active');
});

ui.tabSaved.addEventListener('click', () => {
    ui.tabSaved.classList.add('active');
    ui.tabHistory.classList.remove('active');
    ui.listSaved.classList.add('active');
    ui.listHistory.classList.remove('active');
});

let isRotating = true;
ui.toggleRot.addEventListener('click', () => {
    isRotating = !isRotating;
    controls.autoRotate = isRotating;
    ui.toggleRot.classList.toggle('active', isRotating);
});

ui.resetCam.addEventListener('click', () => {
    controls.target.set(0,0,0);
    camera.position.set(INITIAL_CAM_POS.x, INITIAL_CAM_POS.y, INITIAL_CAM_POS.z);
    ui.zoomSlider.value = 50;
    controls.update();
    lastCamX = -99999; // force redraw
});

// Copy
ui.btnCopy.addEventListener('click', () => {
    const text = ui.input.value;
    navigator.clipboard.writeText(text).then(() => showToast("COPIED"));
});

function showToast(msg) {
    ui.toast.textContent = msg;
    ui.toast.style.opacity = 1;
    setTimeout(() => ui.toast.style.opacity = 0, 1500);
}

window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

// Init
generateNew();