

### Migrate registry files

In the past years, we store the key-value registry (for experimental parameters) in the binary one file per key format (Delphi format), which spend a long time on sorting files when we need to copy-paste the full folder.

We can migrate the registry from the Delphi format into the new SQLite format. 

- Download scalabrad from [bintray](https://bintray.com/labrad/generic/scalabrad#files)

  for example, scalabrad-0.8.3

- Open the directory `.\scalabrad-0.8.3\bin`, start command console (CMD) and use the following: 

```
labrad-migrate-registry file:///M:/Registry-old-delphi?format=delphi file:///M:/Registry-new
```

Note: the paths 'M:/Registry-old-delphi' and 'M:/Registry-new' are just an example, you should change it into yours. 

- After running, your old files will be migrated into the SQLite format. 



```
Migrate registry between managers and formats.

Usage: labrad-migrate-registry [OPTIONS] src [dst]

OPTIONS

-v
--verbose  Print progress information during the migration

-h
--help     Print usage information and exit

PARAMETERS

src  Location of source registry. Can be labrad://[<pw>@]<host>[:<port>] to
     connect to a running manager, or file://<path>[?format=<format>] to load
     data from a local file. If the path points to a directory, we assume it is
     in the binary one file per key format; if it points to a file, we assume
     it is in the new SQLite format. The default format can be overridden by
     adding a 'format=' query parameter to the URI, with the following options:
     binary (one file per key with binary data), delphi (one file per key with
     legacy delphi text-formatted data), sqlite (single sqlite file for entire
     registry).

dst  Location of destination registry. Can be labrad://[<pw>@]<host>[:<port>]
     to send data to a running manager, or file://<path>[?format=<format>] to
     write data to a local file. If writing to a file, the URI will be
     interpreted as for the src param. If sending the data to a running
     manager, the data will be stored in whatever format that manager uses. If
     not specified, we traverse the source registry but do not transfer the
     data to a new location. This can be used as a dry run to verify the
     integrity of the source registry before migration.
```



#### About Registry-editor

After migrating the registry, we do not need to use the old Delphi `RegistryEditor.exe`. 

In the recent years, there are many easy-to-use editors for SQLite format.

