const $=s=>document.querySelector(s),B=[[12,'Good','#00e400'],[35.4,'Moderate','#ffe600'],[55.4,'Unhealthy for Sensitive','#ff9a00'],[150.4,'Unhealthy','#ff4d4d'],[250.4,'Very Unhealthy','#a64ca6'],[1e9,'Hazardous','#7e0023']];
const band=v=>B.find(b=>v<=b[0]),f1=v=>Number(v).toFixed(1);
const ADV={General:['Good','Normal activity is fine.'],Child:['Moderate','Limit long outdoor play.'],Asthma:['Moderate','Keep your inhaler handy; avoid exertion outdoors.'],Athlete:['Moderate','Move hard workouts indoors.'],Outdoor:['Moderate','Wear a fitted N95 mask on site.']};
function advice(p,v){const lv=v<=35.4?0:v<=55.4?1:v<=150.4?2:3;const T={General:['Normal activity is fine.','Sensitive people should take it easy.','Reduce time outdoors; wear a mask.','Stay indoors; avoid all outdoor activity.'],Child:['Outdoor play is fine.','Keep outdoor play short.','Keep children indoors; use a purifier.','Keep children indoors; seal windows.'],Asthma:['Keep your inhaler with you.','Avoid exertion outdoors; keep medication ready.','Stay indoors and follow your action plan.','Stay indoors; contact a doctor if symptomatic.'],Athlete:['Train as usual.','Lower the intensity of outdoor workouts.','Move training indoors.','No outdoor training today.'],Outdoor:['No special precaution.','Take breaks in clean air.','Wear a fitted N95 mask for the whole shift.','Avoid work outdoors where possible; N95 mandatory.']};return T[p][lv]}
function chart(cv,hrs){const r=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*r;cv.height=h*r;const c=cv.getContext('2d');c.scale(r,r);const m={l:44,r:12,t:14,b:28},W=w-m.l-m.r,H=h-m.t-m.b;
const mx=Math.max(70,...hrs.map(x=>Math.max(x.hi,x.actual||0)))*1.08,X=i=>m.l+W*i/(hrs.length-1),Y=v=>m.t+H*(1-v/mx),mu=getComputedStyle(document.body).getPropertyValue('--mut');
c.font='12px system-ui';c.fillStyle=mu;c.strokeStyle='#8fa0c033';
for(let v=0;v<=mx;v+=Math.ceil(mx/5/10)*10){c.beginPath();c.moveTo(m.l,Y(v));c.lineTo(w-m.r,Y(v));c.stroke();c.fillText(v,6,Y(v)+4)}
c.setLineDash([5,4]);c.strokeStyle='#ff5d6c';c.beginPath();c.moveTo(m.l,Y(55.4));c.lineTo(w-m.r,Y(55.4));c.stroke();c.setLineDash([]);c.fillStyle='#ff5d6c';c.fillText('Unhealthy 55.4',m.l+4,Y(55.4)-5);
c.fillStyle='#2dd4bf33';c.beginPath();hrs.forEach((x,i)=>c[i?'lineTo':'moveTo'](X(i),Y(x.hi)));for(let i=hrs.length-1;i>=0;i--)c.lineTo(X(i),Y(hrs[i].lo));c.fill();
const L=(k,col,wd)=>{c.strokeStyle=col;c.lineWidth=wd;c.beginPath();let s=0;hrs.forEach((x,i)=>{if(x[k]==null)return;c[s?'lineTo':'moveTo'](X(i),Y(x[k]));s=1});c.stroke()};
L('actual','#e8eefc',1.6);L('pred','#2dd4bf',3);c.fillStyle=mu;hrs.forEach((x,i)=>{if(i%4==3)c.fillText(x.t.slice(11,16),X(i)-14,h-8)})}
async function dash(){const D=await (await fetch('data/forecast.json')).json();let city=Object.keys(D.cities)[0],prof='General';
$('#issued').textContent=D.issued;const tabs=$('#tabs');
const draw=()=>{const C=D.cities[city],now=C.now,b=band(now);document.querySelectorAll('#tabs button').forEach(x=>x.classList.toggle('on',x.textContent==city));
$('#now').innerHTML=`<div class=mut>PM2.5 now</div><div class=big>${f1(now)}</div><span class=pill style="background:${b[2]}">${b[1]}</span>`;
const mxp=C.hours.reduce((a,x)=>x.pred>a.pred?x:a);const bb=band(mxp.pred);$('#peak').innerHTML=`<div class=mut>Forecast peak (next 24h)</div><div class=big>${f1(mxp.pred)}</div><span class=pill style="background:${bb[2]}">${bb[1]}</span><div class=mut>at ${mxp.t.slice(11,16)}, 80% range ${f1(mxp.lo)}–${f1(mxp.hi)}</div>`;
$('#p').innerHTML=Object.entries(C.p_exceed).map(([h,p])=>`<div>${h}h ahead: <b>${Math.round(p*100)}%</b><div class=bar><i style="width:${p*100}%"></i></div></div>`).join('');
$('#adv').textContent=advice(prof,mxp.hi>mxp.pred?(mxp.pred+mxp.hi)/2:mxp.pred);chart($('#cv'),C.hours)};
Object.keys(D.cities).forEach(c=>{const b=document.createElement('button');b.textContent=c;b.onclick=()=>{city=c;draw()};tabs.append(b)});
document.querySelectorAll('[data-prof]').forEach(b=>b.onclick=()=>{prof=b.dataset.prof;draw()});draw();addEventListener('resize',draw)}
async function evalp(){const M=await (await fetch('data/metrics.json')).json();const P=M.pooled_cv;
$('#kpis').innerHTML=[['24h MAE improvement vs persistence',P[24].skill_B_vs_persistence_pct+'%'],['24h MAE (µg/m³)',P[24].predB.MAE+' vs '+P[24].persist.MAE],['Episode F1 at 6h',M.episode_classifier[6].robust_F1],['Held-out months',M.folds.length]].map(k=>`<div class=card><div class=mut>${k[0]}</div><div class=big>${k[1]}</div></div>`).join('');
const cols=['persist','snaive','mean24','clim','predA','predB'],nm=['Persistence','Seasonal-naive','24h mean','Climatology','Model A','Model B'];
$('#tbl').innerHTML='<tr><th>Horizon</th>'+nm.map(n=>`<th>${n}</th>`).join('')+'</tr>'+M.horizons.map(h=>{const r=P[h];return`<tr><td>${h}h</td>`+cols.map(c=>`<td class="${c=='predB'?'win':''}">${r[c].MAE}</td>`).join('')+'</tr>'}).join('')}
const pg=document.body.dataset.page;document.querySelector(`nav a[data-p=${pg}]`)?.classList.add('on');pg=='dash'&&dash();pg=='eval'&&evalp();
