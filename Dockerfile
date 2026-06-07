FROM ruby:3.3

WORKDIR /srv/jekyll

COPY Gemfile Gemfile.lock ./

RUN bundle install