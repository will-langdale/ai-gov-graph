# GOV.UK Content API acquisition research

This note records the public GOV.UK interfaces relevant to issue #3. It was
checked on 16 August 2026.

## Recommended acquisition flow

1. Enumerate candidate GOV.UK paths with the Search API:
   `GET https://www.gov.uk/api/search.json`.
2. Request `fields=link`, `fields=content_id`, and
   `fields=public_timestamp`. Apply the configured department filter. For
   example, use `filter_organisations=department-for-business-and-trade`.
   `title` is useful for diagnostics. `link` is usually the content item's base
   path. Validate it before retrieval. A link can be absolute or lack a leading
   slash. [Search API guide](https://docs.publishing.service.gov.uk/repos/search-api/using-the-search-api.html), [field definitions](https://github.com/alphagov/search-api/blob/main/config/schema/field_definitions.json)
3. Fetch each returned path with
   `GET https://www.gov.uk/api/content/<path-without-leading-slash>` and retain
   the successful response body as received. The Content API requires no
   authentication. [Content API quick start](https://content-api.publishing.service.gov.uk/)
4. Parse a separate copy only to derive manifest metadata. Do not reserialise
   the downloaded body. The Content API is schema-flexible. `details` varies by
   content type. [Content Store API](https://docs.publishing.service.gov.uk/repos/content-store/content-store-api.html)
5. Build the final manifest order explicitly from a stable key such as
   `base_path`. Do not depend on search result or filesystem order. Record the
   effective query and every source path. This shows when a changed search index
   changes an acquisition.

## Search API

The Search API is public at `https://www.gov.uk/api/search.json`. It validates
parameters strictly. It returns HTTP 422 for unknown parameters and invalid
values. [Search API guide](https://docs.publishing.service.gov.uk/repos/search-api/using-the-search-api.html)

Use `count` and `start` for pagination:

- `count` defaults to 10 and is capped at 1,500.
- `start` is a zero-based offset. An offset beyond the result set returns an
  empty page rather than an error.
- Set `start` to the number of results already consumed. Keep the other
  parameters unchanged. Stop on an empty `results` array. The current GOV.UK
  Search API implementation also presents `total`. Use it as a check, not as
  the only stopping condition. The index can change during enumeration.
  [Search API guide](https://docs.publishing.service.gov.uk/repos/search-api/using-the-search-api.html), [result-set presenter](https://github.com/alphagov/search-api/blob/main/lib/search/presenters/result_set_presenter.rb)

The current response is an object with `results`, `total`, `start`, and search
fields. Each result contains requested fields and implementation metadata. Do
not reject unexpected fields. The current implementation prefixes a relative
`link` with `/`. The indexed field can also hold a non-path URL. Retrieve only
valid GOV.UK-relative paths. [result-set presenter](https://github.com/alphagov/search-api/blob/main/lib/search/presenters/result_set_presenter.rb), [result presenter](https://github.com/alphagov/search-api/blob/main/lib/search/presenters/result_presenter.rb), [field definitions](https://github.com/alphagov/search-api/blob/main/config/schema/field_definitions.json)

Relevant query options:

- `filter_organisations=<slug>` limits results to an organisation. Repeating
  one filter accepts any supplied value for that field; filter groups for
  different fields are combined. `reject_organisations` excludes a slug.
- `filter_format` and `filter_content_store_document_type` can narrow content
  types if the experiment needs it.
- `order=<field>` selects an allowed sort field, with `-` for descending order.
  Without `q`, the default is recent popularity, so it is unsuitable as the
  source of a reproducible corpus order.
- `fields=link` may be repeated in the usual query-string form when more
  metadata is needed. [Search API guide](https://docs.publishing.service.gov.uk/repos/search-api/using-the-search-api.html)

## Content API

The Content API server is `https://www.gov.uk/api/content`; its required path
parameter has no leading slash. For example, the GOV.UK path
`/foreign-travel-advice/thailand` is requested as
`/api/content/foreign-travel-advice/thailand`. [Content API reference](https://content-api.publishing.service.gov.uk/reference.html)

On success, HTTP 200 returns a ContentItem JSON object. Common metadata include
`base_path`, `content_id`, `locale`, `document_type`, `schema_name`,
`public_updated_at`, `updated_at`, `details`, and `links`. `content_id` combined
with `locale` identifies an individual piece of content. `updated_at` changes
when the item changes, including dependent-link changes; `public_updated_at`
only changes for a major published edition. Neither is an immutable content
revision identifier. [Content API reference](https://content-api.publishing.service.gov.uk/reference.html)

For a source-document version, record at least the requested path, response
`base_path`, `content_id`, `locale`, `updated_at`, and a SHA-256 digest of the
unmodified response bytes. The digest makes the retained artefact itself the
unambiguous version even where metadata semantics differ between schemas.

Content API status handling:

- 200: retain the exact JSON body and manifest entry.
- 303: the requested path is a route whose content lives at another base path.
  Follow the redirect deliberately, then record both the requested and resolved
  paths to expose de-duplication.
- 404: no content exists at the path.
- 410: content is no longer available at the path.

The API has a 10 requests-per-second per-client limit; requests over it may not
be processed and can time out. Rate-limit the downloader below that limit and
surface timeouts, non-200 terminal responses, malformed JSON, and incomplete
enumeration as acquisition failures. [Content API reference](https://content-api.publishing.service.gov.uk/reference.html), [Content API rate limit](https://content-api.publishing.service.gov.uk/)

## Implementation consequences for issue #3

- Keep search enumeration, download results, and the explicit manifest separate
  in the lineage. A successful manifest should be written only after every
  selected path has a verified artefact.
- Preserve HTTP failure context: path, URL, status or exception, and message.
  Do not convert a 303, 404, 410, timeout, or invalid response into a silently
  omitted document.
- The Content API is beta and can change. Store the acquisition timestamp and
  the exact request inputs alongside the artefact identities. [Content API overview](https://content-api.publishing.service.gov.uk/)
