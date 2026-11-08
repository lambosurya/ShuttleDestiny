// ShuttleDestiny model runner: evaluates the exported gradient-boosted trees
// (site/data/model.json) in the browser. Mirrors scikit-learn's
// HistGradientBoostingClassifier prediction logic exactly.
(function (root) {
  "use strict";

  function inBitset(bits, value) {
    var v = value | 0;
    if (v < 0 || v >= bits.length * 32) return false;
    return ((bits[v >> 5] >>> (v & 31)) & 1) === 1;
  }

  function ShuttleModel(spec) {
    this.spec = spec;
    this.classes = spec.classes;
    // scikit-learn re-encodes categorical columns to 0..n-1 using the
    // values it saw in training; unseen values are treated as missing.
    this.catMaps = spec.category_maps.map(function (codes) {
      var map = {};
      codes.forEach(function (code, i) { map[code] = i; });
      return map;
    });
  }

  ShuttleModel.prototype.rawTree = function (tree, x) {
    var node = 0;
    while (!tree.leaf[node]) {
      var f = tree.f[node];
      var value = x[f];
      var missing = value === null || value === undefined || Number.isNaN(value);
      var goLeft;
      if (missing) goLeft = tree.m[node] === 1;
      else if (tree.c[node]) goLeft = inBitset(tree.bits[tree.b[node]], value);
      else goLeft = value <= tree.t[node];
      node = goLeft ? tree.l[node] : tree.r[node];
    }
    return tree.v[node];
  };

  // inputs: object keyed by column name. Shot names and player names are
  // given as text; positions in metres. Returns [{shot, p}] sorted by p.
  ShuttleModel.prototype.predict = function (inputs) {
    var spec = this.spec;
    var x = spec.columns.map(function (col) {
      var v = inputs[col];
      if (col === "receiver") {
        var i = spec.players.indexOf(v);
        return i >= 0 ? i : NaN;
      }
      if (typeof v === "string") {
        var j = spec.vocab.indexOf(v);
        return j >= 0 ? j : NaN;
      }
      return v === undefined || v === null ? NaN : Number(v);
    });
    var self = this;
    for (var c = 0; c < spec.categorical_count; c++) {
      var code = self.catMaps[c][x[c]];
      x[c] = code === undefined ? NaN : code;
    }
    var k = spec.n_classes;
    var raw = spec.baseline.slice();
    for (var t = 0; t < spec.trees.length; t++) {
      raw[t % k] += this.rawTree(spec.trees[t], x);
    }
    var max = Math.max.apply(null, raw);
    var exps = raw.map(function (r) { return Math.exp(r - max); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);
    return this.classes
      .map(function (shot, i) { return { shot: shot, p: exps[i] / sum }; })
      .sort(function (a, b) { return b.p - a.p; });
  };

  root.ShuttleModel = ShuttleModel;
  if (typeof module !== "undefined") module.exports = ShuttleModel;
})(typeof window !== "undefined" ? window : globalThis);
