=======================================================================
NIH NLM NCBI PubMed Central (PMC) Article Datasets - 
Full-Text Biomedical and Life Sciences Journal Articles on AWS
=======================================================================

1. OVERVIEW
-----------
PubMed Central (PMC) is a free full-text archive of biomedical and life
sciences journal articles at the U.S. National Institutes of Health's
National Library of Medicine (NIH/NLM). The PMC Article Datasets include
full-text articles archived in PMC and made available under license terms
that allow for text mining and other types of secondary analysis and reuse.


2. HOW TO CITE
--------------
NIH NLM NCBI PubMed Central (PMC) Article Datasets - Full-Text Biomedical
and Life Sciences Journal Articles on AWS was accessed on DATE
from https://registry.opendata.aws/ncbi-pmc.


3. TERMS AND CONDITIONS
-----------------------
Accessing PMC data on the PMC Cloud Service indicates your acceptance of the
following Terms and Conditions. No charges, usage fees or royalties are paid
to NLM for these data.

PMC Specific Terms:

- Articles available from PMC are provided by the respective publishers 
  or authors. 
- Articles in PMC usually include an explicit copyright statement. See the
  PMC Copyright notice for more information:
  https://pmc.ncbi.nlm.nih.gov/about/copyright/.

General Terms and Conditions:

- Users of the data agree to:
  - acknowledge NLM as the source of the data in a clear and conspicuous
    manner,
  - NOT use the PubMed Central wordmark or the PMC logo in association or in
    connection with user's or any other party's product or service.
  - NOT adopt, use, or seek to register any mark or trade name confusingly
    similar to or suggestive of the PubMed Central wordmark or PMC logo
  - NOT to indicate or imply that NLM/NIH/HHS has endorsed its
    products/services/applications.
  
- Users who redistribute the data (services, products or raw data) agree to:
  - only distribute data that are licensed for redistribution,
  - maintain the most current version of all distributed data, or
  - make known in a clear and conspicuous manner that the
    products/services/applications do not reflect the most current/accurate
    data available from NLM.

- These data are produced with a reasonable standard of care, but NLM makes
  no warranties express or implied, including no warranty of merchantability
  or fitness for particular purpose, regarding the accuracy or completeness
  of the data. Users agree to hold NLM and the U.S. Government harmless from
  any liability resulting from errors in the data. NLM disclaims any
  liability for any consequences due to use, misuse, or interpretation of
  information contained or not contained in the data.

- NLM does not provide legal advice regarding copyright, fair use, or other
  aspects of intellectual property rights. See the NLM Copyright page:
  https://www.nlm.nih.gov/web_policies.html#copyright and the PMC Copyright
  notice: https://pmc.ncbi.nlm.nih.gov/about/copyright/

- NLM reserves the right to change the type and format of its machine-readable
  data. NLM will take reasonable steps to inform users of any changes to the
  format of the data before the data are distributed via the announcement
  section or subscription to email and RSS updates.

- The PubMed Central wordmark is a registered trademark of the U.S.
  Department of Health and Human Services (HHS). Unauthorized use of this
  mark is strictly prohibited.


4. STRUCTURE OF THE DATASET
---------------------------

An article in PMC may have more than one version associated with the same
Accession ID (PMCID), for example an author manuscript and a published
version. The dataset is organized by article version. It includes the
following components:

1. Objects for each of the roughly 8 million PMC article versions are
   collected under a prefix named by the PMC Accession ID and the article
   version number, for example `PMC13901.1`. The prefix includes:
    - an XML file for the article version, encoded according to the latest
      version of ANSI/NISO Z39.96-2015 JATS, and named corresponding to the
      parent prefix, for example `PMC13901.1.xml`,
    - a plain text version of the article, extracted from the XML and also
      named by article version, for example `PMC13901.1.txt`,
    - a JSON object listing core metadata (see below), named by the parent
      prefix, for example `PMC13901.1.json`, and,
    - when permissible by the publishers' licenses, additional media files,
      namely:
      - the PDF file of the article, and
      - images and supplementary data files. 

2. The JSON metadata objects are additionally collected under a `metadata`
   prefix. The JSON includes the following properties:
    - pmcid: the PubMed Central Accession ID
    - version: the article version number
    - pmid: the PubMed ID
    - doi: the Digital Object Identifier
    - title: the article title,
    - citation: the journal citation,
    - is_pmc_openaccess: whether the article version is part of the PMC
      Open Access Subset (https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
    - is_manuscript: whether the article version is an author manuscript
      (https://pmc.ncbi.nlm.nih.gov/about/authorms/),
    - is_historical_ocr: whether the article version is part of the
      historical back-scanning project
      (https://pmc.ncbi.nlm.nih.gov/about/scanning/)
    - is_retracted: whether the article version has been retracted,
    - license_code: a code for the license. This code
      - either corresponds to the Creative Commons license codes
        (https://creativecommons.org/share-your-work/cclicenses/),
      - or is set to 'TDM' for author manuscripts where the full text
        is available for text mining, and where the full text may
        also be used consistent with the principles of fair use under
        the copyright law.
    - xml_url: the S3 URL to the article XML. All S3 URLs in the JSON
      include the MD5 digest of the object in form of a URL parameter, md5.
    - pdf_url: the S3 URL to the article PDF, if available,
    - media_urls: a list of S3 URL to images and supplementary data
      files, if available,
    - text_url: the S3 URL to the plain text version.

3. An Amazon S3 inventory in CSV format is located at
   `s3://pmc-oa-opendata/inventory-reports/pmc-oa-opendata/metadata/`
   The inventory contains the following fields:
    - Bucket name - always `pmc-oa-opendata`
    - Key -  the object key for the JSON metadata object.
    - ETag - the entity tag or checksum of the JSON metadata object. The
      JSON contains the MD5 checksum of each object belonging to the article
      version. Therefore, a change to any of the objects will reflect in a
      changed ETag.
    - Last modified date - the object creation date or the last modified
      date of the JSON metadata file, whichever is the latest. The JSON
      contains the MD5 checksum of each object belonging to the article
      version. Therefore, a change to any of the objects will reflect in a
      changed Last modified date.

Here is a schematic overview of the bucket with only two article versions: 

  s3://pmc-oa-opendata/
  |-- PMC10009416.1
  |   |-- NPR2-43-85-g001.jpg
  |   |-- NPR2-43-85-s001.xlsx
  |   |-- PMC10009416.1.json
  |   |-- PMC10009416.1.pdf
  |   |-- PMC10009416.1.txt
  |   `-- PMC10009416.1.xml
  |-- PMC12788873.1
  |   |-- PMC12788873.1.json
  |   |-- PMC12788873.1.txt
  |   `-- PMC12788873.1.xml
  |-- metadata
  |   |-- PMC10009416.1.json
  |   `-- PMC12788873.1.json
  |-- inventory-reports/
  `-- README.txt


5. DATA ACCESS
-------------------------

The following demonstrates how to access the dataset using
the AWS command line interface. 

For anonymous access, add `--no-sign-request` to all `aws` commands.

5.1 ACCESSING ARTICLE DATA

1. To list the contents of the bucket:
  
    $ aws s3 ls s3://pmc-oa-opendata/

    Caution! This will print about 8 million prefixes. 
    To see a list of every article version, refer also 
    to the inventory in the next section.

2. To list the objects belonging to a specific article version: 

    $ aws s3 ls s3://pmc-oa-opendata/PMC10009402.1/
    
    Or via HTTPS with `curl` or in a browser:
    
    $ curl "https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC10009402.1/"
    
       Note that an article-version name such as PMC10009402.1/ is an 
       S3 key prefix, not an object or directory. Opening that URL 
       directly returns NoSuchKey; use the ListObjectsV2 URL as shown
       above to list the objects under the prefix.

3. To download a specific object:

    $ aws s3 cp s3://pmc-oa-opendata/PMC10009402.1/PMC10009402.1.xml .
    
    Or:
    
    $ curl "https://pmc-oa-opendata.s3.amazonaws.com/PMC10009402.1/PMC10009402.1.xml"

4. To download all objects belonging to a specific article version: 

    $ aws s3 cp --recursive s3://pmc-oa-opendata/PMC10009402.1 PMC10009402.1

5. To see all versions belonging to a PMCID

    $ aws s3api list-objects-v2 --bucket pmc-oa-opendata --prefix "PMC11370360." \
    --delimiter "/" --query "CommonPrefixes[].Prefix" --output "text"
   
    Or:
   
    $ curl -s "https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC11370360.&delimiter=/" | \
    grep -oP '(?<=<CommonPrefixes><Prefix>)[^<]+'

    outputs:
      PMC11370360.1
      PMC11370360.2


5.2 ACCESSING THE INVENTORY

The Amazon S3 inventory, in CSV format, is regenerated once a day.

1. To find the latest version

    $  aws s3 ls s3://pmc-oa-opendata/inventory-reports/pmc-oa-opendata/metadata/ \
       | awk '{print $2}' | grep -v hive | grep -v data | sort | tail -1  
    
    outputs, for example: 

        2026-01-18T01-00Z/

2. To read the manifest for a specific inventory version and retrieve the CSV path: 

    $ aws s3 cp s3://pmc-oa-opendata/inventory-reports/pmc-oa-opendata/metadata/2026-01-18T01-00Z/manifest.json - \
      | jq '.files[].key'
    
    outputs:

        "inventory-reports/pmc-oa-opendata/metadata/data/8f5c5e61-02eb-4ee2-b846-3fa7ae63dc17.csv.gz"

3. To then download and read the inventory: 

     $ aws s3 cp s3://pmc-oa-opendata/inventory-reports/pmc-oa-opendata/metadata/data/8f5c5e61-02eb-4ee2-b846-3fa7ae63dc17.csv.gz .
     $ gunzip 8f5c5e61-02eb-4ee2-b846-3fa7ae63dc17.csv.gz 
     $ head -3 8f5c5e61-02eb-4ee2-b846-3fa7ae63dc17.csv

     Outputs, for example:
        "pmc-oa-opendata","metadata/PMC10000002.1.json","2026-01-08T11:47:59.000Z","f25d9c5f5b85338f62156faeedfc1e95"
        "pmc-oa-opendata","metadata/PMC10000003.1.json","2026-01-08T11:55:14.000Z","a47959089c1fd488b91004eeb06adfa9"
        "pmc-oa-opendata","metadata/PMC10000005.1.json","2026-01-08T12:10:34.000Z","22ed20dfdd305479a4ade2d573514b3d"

Inventory reports are retained for 30 days and then automatically removed. 
If you need to archive a report long-term, download or copy it to another
location before it expires.


5.3 USING AN ESEARCH-S3 PIPELINE

The "ESearch" Entrez Programming Utility responds to a text query
with the list of matching PMC Identifiers, in integer form. Refer to
https://www.ncbi.nlm.nih.gov/books/NBK25497/ for API documentation.

Appending the filters "(open_access[Filter] OR author_manuscript[Filter])"
to the query will limit results to articles available in this dataset.

The articles can then be mapped to S3 prefixes and retrieved.

1. Run an eSearch, for example for alzheimers:

    $  curl -s "https://eutils.ncbi.nlm.nih.gov/eutils/esearch.fcgi?db=pmc&term=alzheimers+AND+(open_access[Filter]+OR+author_manuscript[Filter])&format=json" \
      | jq '.esearchresult.idlist'

    This returns an idlist:
      [
        "12810641",
        "12810747",
        ...
      ]

2. Find the articles version(s) associated with the PMCID, prefixing 'PMC' and adding the period:

    $ aws s3api list-objects-v2 --bucket pmc-oa-opendata --prefix "PMC12810641." \
    --delimiter "/" --query "CommonPrefixes[].Prefix" --output "text"

    returns:
       PMC12810641.1/

3. Download the respective article version:
    
    $ aws s3 cp --recursive s3://pmc-oa-opendata/PMC12810641.1/ PMC12810641.1/

For a complete list of PMC search fields, see
https://pmc.ncbi.nlm.nih.gov/about/userguide/. Here are a few example
queries:

- Find articles added on a specific day: 
  2026/01/18[pmcrdat] AND "(open_access[Filter] OR author_manuscript[Filter])" 

- Find articles added during a specific date range:
  2026/1/14:2026/1/31[pmcrdat] AND (open_access[Filter] OR author_manuscript[Filter])

- Find articles that permit commercial reuse:
  (
    cc0_license[Filter] OR cc_by_license[Filter] OR 
    cc_by-sa_license[Filter] OR cc_by-nd_license[Filter] 
   ) AND (open_access[Filter] OR author_manuscript[Filter]) 

- Find articles that only permit non-commercial reuse:
  (
    cc_by-nc_license[Filter] OR cc_by-nc-nd_license[Filter]
    OR cc_by-nc-sa_license[Filter]
   ) AND (open_access[Filter] OR author_manuscript[Filter])


6. VERSIONING AND UPDATE FREQUENCY
----------------------------------

Article versions are updated continuously. Updates include:
- addition of new article versions
- the update of one or all objects belonging to an existing article version
- in rare cases, the removal of an article version

The Amazon inventory is updated on a daily basis. This means the inventory
lags the bucket state and may not cover the most recent changes.


7. RELATED RESOURCES
--------------------

For additional information, please see:

- NIH NCBI PubMed Central (PMC) Article Datasets - Full-Text 
  Biomedical and Life Sciences Journal Articles on AWS:
  https://registry.opendata.aws/ncbi-pmc/
- PMC Open Access Subset: 
  https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- Accessing PMC Article Datasets Using Amazon Web Services
  https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
- PMC User Guide: 
  https://pmc.ncbi.nlm.nih.gov/about/userguide/
- Entrez® Programming Utilities Help: 
  https://www.ncbi.nlm.nih.gov/books/NBK25501/


8. CONTACT AND SUPPORT
-----------------------

All questions should be directed to: pubmedcentral@ncbi.nlm.nih.gov 
