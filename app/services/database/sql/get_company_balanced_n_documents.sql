WITH BaseData AS (
    SELECT
        dh.Id AS belgeID,
        gd.Id AS belgeTipi,
        feb.Id AS firmaID
    FROM DosyaDetay dd
    JOIN DosyaHareket dh ON dd.DosyaHareketId = dh.Id
    LEFT JOIN ANOrder a ON dd.IslemId = a.Id
    LEFT JOIN GrupDetay gd ON dh.GrupDetayId = gd.Id
    LEFT JOIN FirmaEvrimBilgileri feb ON a.EvrimFirmaId = feb.Id
    WHERE
        dd.Deleted = 0
        AND dd.IslemTuru IN (5,6)
        AND dh.Silindi = 0
        AND gd.Id IN ({allowed_ids})
        AND (feb.Id IS NULL OR feb.Id != '631BD182-E780-4173-8E98-AC6695D301B6')
),

FirmaCounts AS (
    SELECT
        belgeTipi,
        firmaID,
        COUNT(*) AS firmaToplam
    FROM BaseData
    WHERE firmaID IS NOT NULL
    GROUP BY belgeTipi, firmaID
),

FirmaSayisi AS (
    SELECT
        belgeTipi,
        COUNT(DISTINCT firmaID) AS firmaAdet
    FROM BaseData
    WHERE firmaID IS NOT NULL
    GROUP BY belgeTipi
),

Limitler AS (
    SELECT
        f.belgeTipi,
        f.firmaID,
        CASE
            WHEN f.firmaToplam < CEILING(1.0 * {n_documents} / fs.firmaAdet)
            THEN f.firmaToplam
            ELSE CEILING(1.0 * {n_documents} / fs.firmaAdet)
        END AS firmaLimit
    FROM FirmaCounts f
    JOIN FirmaSayisi fs
      ON f.belgeTipi = fs.belgeTipi
),

Ranked AS (
    SELECT
        b.belgeID,
        b.belgeTipi,
        b.firmaID,
        ROW_NUMBER() OVER (
            PARTITION BY b.belgeTipi, b.firmaID
            ORDER BY NEWID()
        ) AS rn,
        l.firmaLimit
    FROM BaseData b
    JOIN Limitler l
      ON b.belgeTipi = l.belgeTipi
     AND b.firmaID = l.firmaID
)

SELECT
    belgeID,
    belgeTipi,
    firmaID
FROM Ranked
WHERE rn <= firmaLimit

UNION ALL

-- Ceza
SELECT TOP ({n_documents})
    belgeID,
    belgeTipi,
    firmaID
FROM BaseData
WHERE firmaID IS NULL
  AND belgeTipi = 13

ORDER BY belgeTipi, firmaID;