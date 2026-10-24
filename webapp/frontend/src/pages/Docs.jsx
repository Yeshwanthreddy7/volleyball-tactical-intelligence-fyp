import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { marked } from "marked";
import { api, repoFile } from "../api.js";
import { GITHUB_FALLBACK } from "../App.jsx";

function resolve(base, href) {
  const parts = (base ? base.split("/") : []).filter((p) => p && p !== ".");
  for (const seg of href.split("/")) {
    if (seg === "..") parts.pop();
    else if (seg && seg !== ".") parts.push(seg);
  }
  return parts.join("/");
}

function render(markdown, base) {
  const renderer = new marked.Renderer();
  const img = renderer.image.bind(renderer);
  const link = renderer.link.bind(renderer);
  const external = (h) => /^(https?:|mailto:|#)/.test(h);
  renderer.image = (tok) => img({ ...tok, href: external(tok.href) ? tok.href : repoFile(resolve(base, tok.href)) });
  renderer.link = (tok) => {
    let href = tok.href;
    if (!external(href)) {
      const path = resolve(base, href.split("#")[0]);
      href = /\.(png|jpe?g|svg|gif)$/i.test(path) ? repoFile(path) : `${GITHUB_FALLBACK}/blob/main/${path}`;
    }
    return link({ ...tok, href }).replace("<a ", '<a target="_blank" rel="noreferrer" ');
  };
  return marked.parse(markdown, { renderer, gfm: true });
}

export default function Docs() {
  const { slug } = useParams();
  const nav = useNavigate();
  const [index, setIndex] = useState([]);
  const [doc, setDoc] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.docs().then(setIndex).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    if (!slug && index.length) nav(`/docs/${index[0].slug}`, { replace: true });
  }, [slug, index, nav]);
  useEffect(() => {
    if (!slug) return;
    setDoc(null);
    api.doc(slug).then(setDoc).catch((e) => setErr(e.message));
  }, [slug]);

  const html = useMemo(() => (doc ? render(doc.markdown, doc.base === "." ? "" : doc.base) : ""), [doc]);

  return (
    <div className="container section">
      <div className="eyebrow">Documentation</div>
      <h1>Project documentation</h1>
      <p className="lead">The project's own write-ups, served directly from the repository so they are always current.</p>
      <div className="docs-layout" style={{ marginTop: 26 }}>
        <nav className="docs-nav">
          {index.map((d) => (
            <Link key={d.slug} to={`/docs/${d.slug}`} className={d.slug === slug ? "active" : ""}>{d.title}</Link>
          ))}
          <a href={GITHUB_FALLBACK} target="_blank" rel="noreferrer">Source code on GitHub ↗</a>
        </nav>
        <div style={{ minWidth: 0 }}>
          {err && <div className="alert err">{err}</div>}
          {!doc && !err && <div className="empty"><span className="spinner" /></div>}
          {doc && (
            <>
              <div className="small faint" style={{ marginBottom: 10 }}>{doc.path}</div>
              <article className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
