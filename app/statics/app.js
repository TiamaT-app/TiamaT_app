/* Affiche le nom du projet courant dans l'en-tête. Le nom est gardé en sessionStorage,
   car toutes les routes ne le transmettent pas au template. */
(function () {
  var CLE = 'tiamat.projet';
  var pastille = document.querySelector('[data-projet]');
  if (!pastille) return;

  var nom = pastille.getAttribute('data-projet');
  try {
    if (nom) sessionStorage.setItem(CLE, nom);
    else nom = sessionStorage.getItem(CLE) || '';
  } catch (e) {
    nom = nom || '';
  }
  if (!nom) return;

  pastille.querySelector('[data-projet-nom]').textContent = nom;
  pastille.hidden = false;
})();

/* Traitements longs : un bouton ou lien avec data-attente affiche le libellé d'attente
   et ne peut être déclenché qu'une fois. */
(function () {
  function marquer(el) {
    if (el.getAttribute('aria-busy') === 'true') return;
    el.dataset.libelleOrigine = el.textContent;
    el.textContent = el.dataset.attente;
    el.setAttribute('aria-busy', 'true');
  }
  document.addEventListener('click', function (e) {
    var lien = e.target.closest('a[data-attente]');
    if (!lien || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (lien.getAttribute('aria-busy') === 'true') { e.preventDefault(); return; }
    marquer(lien);
  });
  document.addEventListener('submit', function (e) {
    var bouton = e.target.querySelector('button[data-attente]');
    if (!bouton) return;
    marquer(bouton);
    setTimeout(function () { bouton.disabled = true; }, 0);
  });
  window.addEventListener('pageshow', function (e) {
    if (!e.persisted) return;
    document.querySelectorAll('[aria-busy="true"]').forEach(function (el) {
      el.textContent = el.dataset.libelleOrigine;
      el.removeAttribute('aria-busy');
      el.disabled = false;
    });
  });
})();

/* Formulaires d'envoi d'images : affiche le nombre d'images choisies et leurs noms. */
(function () {
  document.querySelectorAll('form[data-envoi]').forEach(function (form) {
    var champ = form.querySelector('input[type="file"]');
    var etat = form.querySelector('[data-selection]');
    var liste = form.querySelector('[data-liste]');
    if (!champ || !etat || !liste) return;
    champ.addEventListener('change', function () {
      var n = champ.files.length;
      etat.textContent = n === 0 ? 'Aucune image choisie' : n + (n === 1 ? ' image choisie' : ' images choisies');
      liste.textContent = '';
      Array.prototype.forEach.call(champ.files, function (fichier) {
        var li = document.createElement('li');
        li.textContent = fichier.name;
        li.title = fichier.name;
        liste.appendChild(li);
      });
      liste.hidden = n === 0;
    });
  });
})();

/* Jeton CSRF : /accueil_projet n'accepte que des envois avec un jeton valide, qui peut expirer
   sur une page ouverte depuis longtemps. Le jeton est relu sur la page d'accueil. */
(function () {
  function jetonFrais() {
    return fetch('/', { credentials: 'same-origin' })
      .then(function (r) { return r.text(); })
      .then(function (html) {
        var jeton = new DOMParser().parseFromString(html, 'text/html').querySelector('input[name="csrf_token"]');
        if (!jeton) throw new Error('jeton introuvable');
        return jeton.value;
      });
  }

  /* Bouton « retour au corpus » : envoie le formulaire avec un jeton frais. */
  document.querySelectorAll('form[data-csrf-auto]').forEach(function (form) {
    var champ = form.querySelector('input[name="csrf_token"]');
    form.addEventListener('submit', function (e) {
      if (champ.value) return;
      e.preventDefault();
      jetonFrais()
        .then(function (jeton) { champ.value = jeton; form.submit(); })
        .catch(function () { window.location.href = '/#projets'; });
    });
  });

  /* Après un envoi, la route ne transmet pas le total du projet : il est relu sur la page
     Corpus en arrière-plan (attribut data-total). En cas d'échec, rien n'est ajouté. */
  var cible = document.querySelector('[data-total-projet]');
  if (!cible) return;
  jetonFrais()
    .then(function (jeton) {
      var corps = new URLSearchParams({ csrf_token: jeton, nom: '', projet_existant: cible.getAttribute('data-projet') });
      return fetch('/accueil_projet', { method: 'POST', body: corps, credentials: 'same-origin' });
    })
    .then(function (r) { return r.text(); })
    .then(function (html) {
      var section = new DOMParser().parseFromString(html, 'text/html').querySelector('[data-total]');
      var total = section ? parseInt(section.getAttribute('data-total'), 10) : 0;
      if (!total) return;
      cible.textContent = ' Le projet contient maintenant ' + total + (total === 1 ? ' image.' : ' images.');
      var galerie = document.querySelector('[data-total-galerie]');
      if (galerie && total > parseInt(galerie.getAttribute('data-affiche'), 10)) galerie.textContent = ' sur ' + total;
    })
    .catch(function () {});
})();

/* Sondage (page de lancement de Label Studio) : interroge l'URL indiquée à intervalle
   régulier, et suit la redirection dès qu'elle a lieu. */
(function () {
  var zone = document.querySelector('[data-sondage]');
  if (!zone) return;
  var url = zone.getAttribute('data-sondage');
  var delai = parseInt(zone.getAttribute('data-sondage-delai'), 10) || 2000;
  var minuteur = setInterval(function () {
    fetch(url).then(function (r) {
      if (r.redirected) { clearInterval(minuteur); window.location.href = r.url; }
    }).catch(function () {});
  }, delai);
})();
