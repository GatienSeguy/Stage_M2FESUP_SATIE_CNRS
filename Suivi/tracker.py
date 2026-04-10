#!/usr/bin/env python3
"""
Stage VB-ΠGDM — Tracker local connecté à Notion
  python tracker.py   → http://localhost:8400
  Ctrl+C pour arrêter
"""
import json,os,sys,webbrowser,datetime,time
from http.server import HTTPServer,BaseHTTPRequestHandler
from urllib.parse import parse_qs
try: import requests
except ImportError: sys.exit("pip install requests")

NOTION_TOKEN = os.environ.get("NOTION_TOKEN","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")
DATABASE_ID  = os.environ.get("NOTION_DB_ID","b2ea9840c64949e391835e31e413425c")
PORT=8400
API="https://api.notion.com/v1"
HDR={"Authorization":f"Bearer {NOTION_TOKEN}","Notion-Version":"2022-06-28","Content-Type":"application/json"}
TH=[("Signal","Signal & Maths","#6366f1"),("Diffusion","Diffusion","#d97706"),
    ("VB","Inférence VB","#059669"),("Python","Code Python","#2563eb"),
    ("Biblio","Biblio & Rédac.","#db2777"),("Module","Module Python","#7c3aed"),
    ("Orga","Orga & Autres","#78716c")]
TH_KEYS=[k for k,_,_ in TH]

def qdb():
    url,pages=f"{API}/databases/{DATABASE_ID}/query",[]
    pl={"sorts":[{"property":"Date","direction":"ascending"}],"page_size":100}
    while True:
        r=requests.post(url,headers=HDR,json=pl)
        if r.status_code==404: sys.exit("❌ 404 — Connecte l'intégration à la page Notion")
        if r.status_code==401: sys.exit("❌ 401 — Token invalide")
        r.raise_for_status();d=r.json();pages.extend(d["results"])
        if not d.get("has_more"):break
        pl["start_cursor"]=d["next_cursor"]
    return pages

def parse(pages):
    out=[]
    for p in pages:
        pr=p["properties"];pid=p["id"]
        def n(k):
            v=pr.get(k,{})
            if v.get("type")=="number":return v.get("number") or 0
            if v.get("type")=="formula":return v.get("formula",{}).get("number") or 0
            return 0
        def t(k):
            v=pr.get(k,{});return "".join(x.get("plain_text","") for x in v.get("title",v.get("rich_text",[])))
        date="";d=pr.get("Date",{})
        if d.get("date"):date=d["date"].get("start","")
        hum="";h=pr.get("Humeur",{})
        if h.get("select"):hum=h["select"].get("name","")
        vals={k.lower():n(k) for k in TH_KEYS}
        total=sum(vals.values())  # calcul local, pas la formule Notion
        out.append(dict(id=pid,jour=t("Jour"),date=date,**vals,
                        total=total,objectifs=t("Objectifs"),notes=t("Notes"),humeur=hum))
    out.sort(key=lambda e:e.get("date",""));return out

def aggregate_by_day(entries):
    """Fusionne les entrées par date, avec objectifs associés au thème dominant."""
    by_date={}
    for e in entries:
        d=e.get("date","")
        if not d: continue
        if d not in by_date:
            by_date[d]=dict(date=d, jour=e["jour"], **{k.lower():0 for k in TH_KEYS}, total=0, humeur="")
            for k in TH_KEYS: by_date[d][f"obj_{k.lower()}"]=[]
        for k in TH_KEYS:
            by_date[d][k.lower()]+=e.get(k.lower(),0)
        by_date[d]["total"]=sum(by_date[d].get(k.lower(),0) for k in TH_KEYS)
        if e.get("humeur") and not by_date[d]["humeur"]:
            by_date[d]["humeur"]=e["humeur"]
        # associer objectifs au thème avec le plus d'heures dans cette entrée
        obj=e.get("objectifs","").strip()
        if obj:
            best_k=max(TH_KEYS, key=lambda k: e.get(k.lower(),0))
            if e.get(best_k.lower(),0)>0:
                by_date[d][f"obj_{best_k.lower()}"].append(obj)
    for d in by_date.values():
        for k in TH_KEYS:
            d[f"obj_{k.lower()}"]=" · ".join(d[f"obj_{k.lower()}"])
    return sorted(by_date.values(), key=lambda x:x["date"])

def notion_write(vals,page_id=None):
    props={}
    for k in TH_KEYS: props[k]={"number":float(vals.get(k,0))}
    props["Objectifs"]={"rich_text":[{"text":{"content":vals.get("Objectifs","")}}]}
    props["Notes"]={"rich_text":[{"text":{"content":vals.get("Notes","")}}]}
    if vals.get("Humeur"): props["Humeur"]={"select":{"name":vals["Humeur"]}}
    else: props["Humeur"]={"select":None}
    if page_id:
        r=requests.patch(f"{API}/pages/{page_id}",headers=HDR,json={"properties":props})
    else:
        d=datetime.date.today()
        props["Jour"]={"title":[{"text":{"content":d.strftime("%a %d/%m")}}]}
        props["Date"]={"date":{"start":d.isoformat()}}
        r=requests.post(f"{API}/pages",headers=HDR,json={"parent":{"database_id":DATABASE_ID},"properties":props})
    r.raise_for_status()

def page(entries):
    days=aggregate_by_day(entries)
    ej=json.dumps(entries,ensure_ascii=False)
    dj=json.dumps(days,ensure_ascii=False)
    tj=json.dumps([dict(key=k,label=l,color=c) for k,l,c in TH])
    today=datetime.date.today().isoformat()
    now=datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    tot=sum(d["total"] for d in days)
    dw=sum(1 for d in days if d["total"]>0)
    avg=f'{tot/dw:.1f}h' if dw else '—'
    return f'''<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VB-ΠGDM Tracker</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'IBM Plex Sans',system-ui,sans-serif;background:#f5f4f0;color:#2a2a2a;padding:28px 20px;max-width:920px;margin:0 auto;line-height:1.5}}
h1{{font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:700}}h1 b{{color:#6366f1}}
.sub{{font-size:12px;color:#999;margin-top:2px}}.lnk{{color:#6366f1;text-decoration:none}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:22px 0 18px}}
.st{{background:#fff;border:1px solid #e8e6e1;border-radius:10px;padding:16px 12px;text-align:center}}
.sv{{font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:700;line-height:1}}
.sl{{font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.07em;margin-top:4px;font-weight:600}}
.card{{background:#fff;border:1px solid #e8e6e1;border-radius:12px;padding:22px;margin-bottom:16px}}
.ch{{font-size:13px;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:14px}}
.gr2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.tb{{background:#faf9f7;border:1px solid #eeece7;border-radius:8px;padding:14px}}
.th{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.tn{{font-size:13px;font-weight:600}}.tt{{font-size:12px;color:#aaa;font-family:'IBM Plex Mono',monospace}}
canvas{{display:block}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:8px 6px;color:#aaa;border-bottom:2px solid #eee;font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.04em}}
td{{padding:7px 6px;border-bottom:1px solid #f2f1ed}}
tr.ck{{cursor:pointer}}tr.ck:hover td{{background:#f8f7f3}}tr.ac td{{background:#eef2ff!important}}
.dot{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle}}
.fs{{background:#fff;border:1px solid #e8e6e1;border-radius:12px;padding:22px;margin-bottom:16px}}
.fr{{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #f5f4f0}}
.fl{{flex:1;font-size:13px;color:#555;font-weight:500}}
.fc{{display:flex;align-items:center;gap:6px}}
.pm{{width:32px;height:32px;border-radius:8px;border:1px solid #ddd;background:#faf9f7;color:#555;font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .12s;user-select:none}}
.pm:hover{{background:#eee;border-color:#bbb}}.pm:active{{background:#ddd;transform:scale(.95)}}
.fv{{font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:600;width:44px;text-align:center}}
.ft2{{display:flex;justify-content:space-between;padding:12px 0 4px;border-top:2px solid #eee;margin-top:8px;font-weight:600;color:#888}}
.ft2 span:last-child{{font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:700;color:#6366f1}}
input[type=text],textarea{{width:100%;border:1px solid #ddd;border-radius:8px;padding:10px 12px;font-size:13px;font-family:inherit;color:#333;background:#faf9f7;resize:vertical;outline:none}}
input:focus,textarea:focus{{border-color:#6366f1;background:#fff}}
select{{border:1px solid #ddd;border-radius:8px;padding:8px 12px;font-size:13px;font-family:inherit;background:#faf9f7;color:#333}}
.btn{{padding:12px 24px;border:none;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;font-family:inherit;transition:all .2s}}
.bp{{background:#6366f1;color:#fff;flex:1}}.bp:hover{{background:#4f46e5}}
.bc{{background:#f0efeb;color:#888;margin-right:8px}}.bc:hover{{background:#e5e4df}}
.btn:disabled{{opacity:.4;cursor:default}}
.msg{{text-align:center;padding:8px;font-size:13px;margin-top:8px;border-radius:6px}}
.ok{{background:#ecfdf5;color:#059669}}.er{{background:#fef2f2;color:#dc2626}}
.badge{{display:inline-block;font-size:10px;font-weight:600;background:#eef2ff;color:#6366f1;padding:2px 8px;border-radius:4px;margin-left:8px}}
#tip{{position:fixed;pointer-events:none;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,.08);z-index:99;display:none;max-width:320px;line-height:1.6}}
#tip .tip-title{{font-weight:700;color:#333;margin-bottom:4px}}
#tip .tip-row{{display:flex;justify-content:space-between;gap:16px}}
#tip .tip-row span:first-child{{color:#888}}
#tip .tip-row span:last-child{{font-family:'IBM Plex Mono',monospace;font-weight:600}}
.bar-wrap{{position:relative}}
</style></head><body>
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px">
<div><h1><b>◈</b> Stage VB-ΠGDM</h1><p class="sub">Tracker local · Notion</p></div>
<p class="sub">{now} · <a href="/" class="lnk">↻ Rafraîchir</a></p></div>
<div class="stats">
<div class="st"><div class="sv" style="color:#6366f1">{tot:.1f}</div><div class="sl">Heures</div></div>
<div class="st"><div class="sv" style="color:#2563eb">{dw}</div><div class="sl">Jours</div></div>
<div class="st"><div class="sv" style="color:#059669">{avg}</div><div class="sl">Moy/j</div></div>
<div class="st"><div class="sv" style="color:#d97706">{len(days)}</div><div class="sl">Entrées</div></div>
</div>

<div class="fs" id="FC">
<div style="display:flex;align-items:center"><div class="ch" id="fT" style="margin-bottom:0">Nouvelle entrée</div><span class="badge" id="fB" style="display:none">Édition</span></div>
<div id="fD" style="font-size:12px;color:#aaa;margin:4px 0 14px">{today}</div>
<div id="fTH"></div>
<div class="ft2"><span>Total</span><span id="fTot">0h</span></div>
<div style="margin-top:12px">
<input type="text" id="fO" placeholder="Objectifs du jour…" style="margin-bottom:8px">
<textarea id="fN" rows="2" placeholder="Notes, blocages, idées…"></textarea>
<div style="margin-top:10px"><label style="font-size:12px;color:#888;margin-right:8px">Humeur</label>
<select id="fH"><option value="">—</option><option value="🟢">🟢 Bien</option><option value="🟡">🟡 Correct</option><option value="🔴">🔴 Difficile</option></select></div></div>
<div style="margin-top:14px;display:flex;align-items:center">
<button class="btn bc" id="cBtn" style="display:none" onclick="cancel()">Annuler</button>
<button class="btn bp" id="sBtn" onclick="save()">Ajouter dans Notion</button></div>
<div id="sMsg"></div></div>

<div class="card"><div class="ch">Heures par jour</div><div class="bar-wrap"><canvas id="bar" height="170"></canvas></div></div>
<div id="tip"></div>
<div class="card"><div class="ch">Courbes par thème</div><div class="gr2" id="sp"></div></div>
<div class="card"><div class="ch">Historique <span style="font-size:10px;font-weight:400;color:#bbb;text-transform:none">— cliquer pour éditer</span></div>
<div style="overflow-x:auto"><table id="T"></table></div></div>

<script>
const E={ej},D={dj},TH={tj},dpr=devicePixelRatio||1;
const fv={{}};TH.forEach(t=>fv[t.key]=0);let eId=null;

// ── Form ──
const fEl=document.getElementById("fTH");
TH.forEach(t=>{{fEl.innerHTML+=`<div class="fr"><div class="dot" style="background:${{t.color}}"></div><div class="fl">${{t.label}}</div><div class="fc"><button class="pm" onclick="adj('${{t.key}}',-0.5)">−</button><div class="fv" id="fv_${{t.key}}">0h</div><button class="pm" onclick="adj('${{t.key}}',0.5)">+</button></div></div>`}});
function adj(k,d){{fv[k]=Math.max(0,Math.round((fv[k]+d)*10)/10);
  const el=document.getElementById("fv_"+k);el.textContent=fv[k]+"h";
  el.style.color=fv[k]>0?TH.find(t=>t.key===k).color:"#ccc";
  document.getElementById("fTot").textContent=TH.reduce((s,t)=>s+fv[t.key],0)+"h"}}
function loadE(e){{eId=e.id;
  TH.forEach(t=>{{fv[t.key]=e[t.key.toLowerCase()]||0;const el=document.getElementById("fv_"+t.key);el.textContent=fv[t.key]+"h";el.style.color=fv[t.key]>0?TH.find(x=>x.key===t.key).color:"#ccc"}});
  document.getElementById("fTot").textContent=TH.reduce((s,t)=>s+fv[t.key],0)+"h";
  document.getElementById("fO").value=e.objectifs||"";document.getElementById("fN").value=e.notes||"";document.getElementById("fH").value=e.humeur||"";
  document.getElementById("fT").textContent=e.jour||e.date;document.getElementById("fD").textContent=e.date;
  document.getElementById("fB").style.display="inline-block";document.getElementById("cBtn").style.display="block";
  document.getElementById("sBtn").textContent="Mettre à jour";document.getElementById("sBtn").disabled=false;
  document.getElementById("sMsg").innerHTML="";
  document.querySelectorAll("tr.ac").forEach(r=>r.classList.remove("ac"));
  const row=document.querySelector(`tr[data-id="${{e.id}}"]`);if(row)row.classList.add("ac");
  document.getElementById("FC").scrollIntoView({{behavior:"smooth",block:"start"}})}}
function cancel(){{eId=null;TH.forEach(t=>{{fv[t.key]=0;const el=document.getElementById("fv_"+t.key);el.textContent="0h";el.style.color="#ccc"}});
  document.getElementById("fTot").textContent="0h";document.getElementById("fO").value="";document.getElementById("fN").value="";document.getElementById("fH").value="";
  document.getElementById("fT").textContent="Nouvelle entrée";document.getElementById("fD").textContent="{today}";
  document.getElementById("fB").style.display="none";document.getElementById("cBtn").style.display="none";
  document.getElementById("sBtn").textContent="Ajouter dans Notion";document.getElementById("sBtn").disabled=false;
  document.getElementById("sMsg").innerHTML="";document.querySelectorAll("tr.ac").forEach(r=>r.classList.remove("ac"))}}
async function save(){{const btn=document.getElementById("sBtn"),msg=document.getElementById("sMsg");
  btn.disabled=true;btn.textContent="Envoi…";msg.innerHTML="";
  const body=new URLSearchParams();TH.forEach(t=>body.append(t.key,fv[t.key]));
  body.append("Objectifs",document.getElementById("fO").value);
  body.append("Notes",document.getElementById("fN").value);
  body.append("Humeur",document.getElementById("fH").value);
  try{{const r=await fetch(eId?"/update/"+eId:"/add",{{method:"POST",body}});
    if(r.ok){{msg.innerHTML=`<div class="msg ok">${{eId?"Mis à jour":"Ajouté"}} ✓ Rechargement…</div>`;
      setTimeout(()=>window.location.href="/",800)
    }}else{{msg.innerHTML=`<div class="msg er">${{await r.text()}}</div>`;btn.disabled=false;btn.textContent="Réessayer"}}
  }}catch(e){{msg.innerHTML=`<div class="msg er">${{e}}</div>`;btn.disabled=false;btn.textContent="Réessayer"}}
}}

// ── Bar chart with Y-axis + tooltip ──
const barRects=[];
!function(){{const c=document.getElementById("bar"),x=c.getContext("2d"),W=c.parentElement.clientWidth-20,H=170;
c.width=W*dpr;c.height=H*dpr;c.style.width=W+"px";c.style.height=H+"px";x.scale(dpr,dpr);
if(!D.length)return;
const pad={{l:40,r:10,t:10,b:24}},cW=W-pad.l-pad.r,cH=H-pad.t-pad.b;
const mx=Math.max(...D.map(e=>TH.reduce((s,t)=>s+(e[t.key.toLowerCase()]||0),0)),1);
const niceMax=Math.ceil(mx);
// Y-axis
const steps=Math.min(niceMax,6);const stepVal=niceMax/steps;
x.strokeStyle="#e5e3de";x.lineWidth=1;
for(let i=0;i<=steps;i++){{
  const yy=pad.t+cH-i*(cH/steps);
  x.beginPath();x.moveTo(pad.l,yy);x.lineTo(W-pad.r,yy);x.stroke();
  x.fillStyle="#aaa";x.font="500 10px IBM Plex Mono";x.textAlign="right";
  x.fillText(Math.round(stepVal*i)+"h",pad.l-6,yy+3);
}}
// Bars
const bw=Math.min(32,(cW-10)/D.length-3);
D.forEach((e,i)=>{{let y=pad.t+cH;const bx=pad.l+8+i*(bw+3);
TH.forEach(t=>{{const v=e[t.key.toLowerCase()]||0;const bh=v/niceMax*cH;
if(bh>0){{x.fillStyle=t.color;x.globalAlpha=.8;x.beginPath();x.roundRect(bx,y-bh,bw,bh,2);x.fill();x.globalAlpha=1;
  barRects.push({{x:bx,y:y-bh,w:bw,h:bh,theme:t.key,dayIdx:i}});
}}y-=bh}});
x.fillStyle="#aaa";x.font="500 9px IBM Plex Mono";x.textAlign="center";
x.fillText((e.date||"").slice(5),bx+bw/2,H-6)}})}}();
// Tooltip
const tip=document.getElementById("tip"),barCanvas=document.getElementById("bar");
barCanvas.addEventListener("mousemove",function(ev){{
  const rect=barCanvas.getBoundingClientRect();
  const mx=ev.clientX-rect.left,my=ev.clientY-rect.top;
  const hit=barRects.find(r=>mx>=r.x&&mx<=r.x+r.w&&my>=r.y&&my<=r.y+r.h);
  if(hit){{
    const day=D[hit.dayIdx];
    tip.style.display="block";
    const tx=Math.min(ev.clientX+14,window.innerWidth-340);
    tip.style.left=tx+"px";tip.style.top=(ev.clientY-10)+"px";
    let html=`<div class="tip-title">${{day.date}} · ${{day.total}}h</div>`;
    TH.forEach(t=>{{const v=day[t.key.toLowerCase()]||0;if(v>0){{
      const isHovered=t.key===hit.theme;
      const style=isHovered?"font-weight:700":"opacity:.55";
      html+=`<div class="tip-row" style="${{style}}"><span style="color:${{t.color}}">⬤ ${{t.label}}</span><span style="color:${{t.color}}">${{v}}h</span></div>`;
      const obj=day["obj_"+t.key.toLowerCase()];
      if(obj && isHovered)html+=`<div style="font-size:11px;color:#555;padding:2px 0 4px 14px;line-height:1.4">${{obj}}</div>`;
    }}}});
    tip.innerHTML=html;
  }}else{{tip.style.display="none"}}
}});
barCanvas.addEventListener("mouseleave",()=>{{tip.style.display="none"}});

// ── Sparklines (sur données agrégées par jour) ──
const sp=document.getElementById("sp");
TH.forEach((t,i)=>{{const d=D.map(e=>e[t.key.toLowerCase()]||0),s=d.reduce((a,b)=>a+b,0);
sp.innerHTML+=`<div class="tb"><div class="th"><span class="tn" style="color:${{t.color}}">${{t.label}}</span><span class="tt">${{s.toFixed(1)}}h</span></div><canvas id="s${{i}}" height="44"></canvas></div>`}});
TH.forEach((t,i)=>{{const cv=document.getElementById("s"+i);if(!cv)return;
const d=D.map(e=>e[t.key.toLowerCase()]||0);if(d.length<2)return;
const x=cv.getContext("2d"),W=cv.parentElement.clientWidth-12,H=44;
cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+"px";cv.style.height=H+"px";x.scale(dpr,dpr);
const mx=Math.max(...d,.5),pts=d.map((v,j)=>[j/(d.length-1)*W,H-3-v/mx*(H-8)]);
const g=x.createLinearGradient(0,0,0,H);g.addColorStop(0,t.color+"25");g.addColorStop(1,t.color+"05");
x.beginPath();x.moveTo(0,H);pts.forEach(p=>x.lineTo(p[0],p[1]));x.lineTo(W,H);x.closePath();x.fillStyle=g;x.fill();
x.beginPath();pts.forEach((p,j)=>j?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]));x.strokeStyle=t.color;x.lineWidth=2;x.lineJoin="round";x.lineCap="round";x.stroke();
const l=pts[pts.length-1];x.beginPath();x.arc(l[0],l[1],3,0,Math.PI*2);x.fillStyle=t.color;x.fill()}});

// ── Table ──
let h="<thead><tr><th>Jour</th><th>Date</th>";TH.forEach(t=>h+=`<th><span class="dot" style="background:${{t.color}}"></span>${{t.key}}</th>`);
h+="<th>Total</th><th></th></tr></thead><tbody>";
[...E].reverse().forEach(e=>{{h+=`<tr class="ck" data-id="${{e.id}}" onclick='loadE(${{JSON.stringify(e).replace(/'/g,"&#39;")}})'>`;
h+=`<td style="font-weight:600">${{e.jour}}</td><td style="color:#999">${{e.date}}</td>`;
TH.forEach(t=>{{const v=e[t.key.toLowerCase()]||0;h+=`<td style="color:${{v>0?t.color:'#ddd'}};font-family:'IBM Plex Mono',monospace;font-weight:500">${{v}}</td>`}});
h+=`<td style="font-weight:700;color:#6366f1;font-family:'IBM Plex Mono',monospace">${{e.total}}</td><td>${{e.humeur}}</td></tr>`}});
document.getElementById("T").innerHTML=h+"</tbody>";
</script></body></html>'''

class H(BaseHTTPRequestHandler):
    entries=[]
    def log_message(self,*a):pass
    def do_GET(self):
        if self.path!="/":self.send_response(302);self.send_header("Location","/");self.end_headers();return
        self.send_response(200);self.send_header("Content-Type","text/html;charset=utf-8");self.end_headers()
        self.wfile.write(page(H.entries).encode())
    def do_POST(self):
        ln=int(self.headers.get("Content-Length",0));body=parse_qs(self.rfile.read(ln).decode())
        vals={k:float(body.get(k,[0])[0]) for k in TH_KEYS}
        vals["Objectifs"]=body.get("Objectifs",[""])[0]
        vals["Notes"]=body.get("Notes",[""])[0]
        vals["Humeur"]=body.get("Humeur",[""])[0]
        try:
            pid=self.path.split("/update/")[1] if self.path.startswith("/update/") else None
            notion_write(vals,pid)
            time.sleep(0.5)  # laisser Notion calculer la formule
            H.entries=parse(qdb())
            print(f"  {'✏️  Mis à jour' if pid else '➕ Ajouté'} — {len(H.entries)} entrées")
            self.send_response(200);self.end_headers();self.wfile.write(b"ok")
        except Exception as e:
            print(f"  ❌ {e}")
            self.send_response(500);self.end_headers();self.wfile.write(str(e).encode())

if __name__=="__main__":
    if NOTION_TOKEN=="COLLE_TON_SECRET_ICI":
        print("\n  1. notion.so/profile/integrations → New integration → copie le Secret")
        print("  2. Dans Notion, page Stage VB-ΠGDM → ⋯ → Connexions → ajoute l'intégration")
        print("  3. tracker.py ligne 14 : NOTION_TOKEN = \"secret_xxx\"\n");sys.exit()
    print("🔄 Sync Notion...")
    try:H.entries=parse(qdb());print(f"✅ {len(H.entries)} entrées")
    except Exception as e:sys.exit(f"❌ {e}")
    srv=HTTPServer(("127.0.0.1",PORT),H)
    print(f"🌐 http://localhost:{PORT}");print("   Ctrl+C pour arrêter\n")
    webbrowser.open(f"http://localhost:{PORT}")
    try:srv.serve_forever()
    except KeyboardInterrupt:print("\n👋")