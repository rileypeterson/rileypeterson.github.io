## Useful commands
`bundle exec jekyll serve --livereload`

## Notes
* One thing I really want for my site is a sidebar. Everyone online is really quickly to make some fancy plugin. Guess I'll be going the long way to figure this one out. 
* I like the default minima theme but I wanted to make a few changes to it and just see what it actually looks like so I just copied that into my root folder: 
    ```
    cp -r /usr/local/lib/ruby/gems/2.7.0/gems/minima-2.5.1/* ./
    ```
* This had the effect of overwriting the README, unfortunately

## Goals
1. First I've got to design what this thing ought to look like. 
    * I'm going to put the "Menu" on the left side
    * It'll have a circle picture in the upper left
    * Below the pic will be icons and links to socials etc.
    * Then below that Home, About, Projects, Writings
    * Projects should have an arrow that when you hover it shows potential projects
      * If you click Projects, it shows tiles with images and title underneath for each project
    