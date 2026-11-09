# Current public PMC distribution route

The legacy NCBI OA lookup returned404. Current official documentation
(https://pmc.ncbi.nlm.nih.gov/tools/oa-service/ and
https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/, read2026-10-25) says the old
service and legacy distribution were retired in August2026. The original
request's description was stale; preserve it as a failed attempt and do not
retry it. The PMC bin download routes returned HTML rather than Office files.

Add only the official anonymous public host pmc-oa-opendata.s3.amazonaws.com
to a new frozen version-three fetch configuration. The public bucket README
documents HTTPS ListObjectsV2 with an article prefix and individual object GET.
List only PMC6123701. (all its versions, max100 objects), then inspect the
observed article-version metadata and explicitly named supplements. This is
local retrieval of public data, not cloud compute or a signed AWS operation.
Retain existing memory/time/disk/byte/request limits; no whole-bucket inventory.

Do not assume version1 exists or is the correct article type. Validate DOI,
PMCID, license, article/manuscript status and asset hashes from returned metadata
before source admission. The precise stored assets determine subsequent requests.
