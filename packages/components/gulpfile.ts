const { src, dest, parallel } = require('gulp')

function copyIcons() {
    return src(['nodes/**/*.{jpg,png,svg}']).pipe(dest('dist/nodes'))
}

function copyModels() {
    return src(['models.json']).pipe(dest('dist'))
}

exports.default = parallel(copyIcons, copyModels)
