# Read original serialized data only. No source pipeline code is executed.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1)
paths <- list.files(args[[1]], pattern="QUANTITIES_(raw|filtered).RData$", recursive=TRUE, full.names=TRUE)
options(digits=17)
for (path in paths) {
  env <- new.env()
  objects <- load(path, envir=env)
  stopifnot(length(objects) == 1)
  obj <- env[[objects[[1]]]]
  stopifnot(inherits(obj, "data.frame"))
  class(obj) <- "data.frame"
  target <- sub(".RData$", ".csv.gz", path)
  con <- gzfile(target, "wt")
  write.table(obj, con, sep=",", row.names=FALSE, col.names=TRUE, na="NA", qmethod="double")
  close(con)
  cat(path, objects[[1]], nrow(obj), ncol(obj), "\n")
}
