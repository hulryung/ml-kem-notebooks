(function () {
  var path = window.location.pathname;
  var parts = path.split('/').filter(Boolean);
  if (parts.length === 0) return;
  var base = '/' + parts[0];
  var rest = path.substring(base.length) || '/';
  var isKo = rest === '/ko' || rest.indexOf('/ko/') === 0;

  var otherUrl, label;
  if (isKo) {
    var stripped = rest.substring(3) || '/';
    otherUrl = base + stripped;
    label = 'English';
  } else {
    otherUrl = base + '/ko' + (rest === '/' ? '/' : rest);
    label = '한글';
  }

  function inject() {
    if (document.getElementById('lang-switcher')) return;
    var el = document.createElement('div');
    el.id = 'lang-switcher';
    el.style.cssText =
      'position:fixed;top:10px;right:12px;z-index:10000;' +
      'background:#fff;padding:6px 12px;border:1px solid rgba(0,0,0,0.15);' +
      'border-radius:6px;font-size:13px;' +
      'box-shadow:0 2px 6px rgba(0,0,0,0.1);';
    el.innerHTML =
      '<a href="' + otherUrl + '" style="color:#333;text-decoration:none;">' +
      '\u{1F310} ' + label + '</a>';
    document.body.appendChild(el);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }

  // Auto-redirect Korean-preferring users landing on the English intro (first visit only)
  var isEnIntro = !isKo && (
    rest === '/' || rest === '/intro.html' || rest === '/index.html'
  );
  if (isEnIntro && !sessionStorage.getItem('_lang_shown')) {
    sessionStorage.setItem('_lang_shown', '1');
    var lang = (navigator.language || navigator.userLanguage || '').toLowerCase();
    if (lang.indexOf('ko') === 0) {
      window.location.replace(base + '/ko/intro.html');
    }
  }
})();
