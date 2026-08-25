const CACHE="radar-futbol-optimizado-v3";
const SHELL=["/","/index.html","/manifest.webmanifest","/icon.svg"];
self.addEventListener("install",e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL))));
self.addEventListener("activate",e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k))))));
self.addEventListener("fetch",e=>{
 const u=new URL(e.request.url);
 if(u.pathname.startsWith("/api/")){e.respondWith(fetch(e.request,{cache:"no-store"}));return;}
 e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));
});