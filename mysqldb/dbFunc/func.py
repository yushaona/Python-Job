import  pymysql as py
from pymysql.cursors import SSCursor,SSDictCursor,Cursor,DictCursor
def HasColumn(cursor,dbName,tableName,columnName):
    "判断字段是否已经存在"
    cursor.execute(" select count(*) as count from information_schema.columns where table_schema='%s' "
                   " and table_name='%s' and column_name='%s' " %(dbName,tableName,columnName))
    data = cursor.fetchone()
    if data[0] == 0:
        return  False
    else:
        return  True


def importEnteCode(db,cursor,dentalID,enteCode):
    "导入金服侠企业号"
    try:
        cursor.execute(" select userid from db_flybear.t_user where dentalid=%s",(dentalID))
        data = cursor.fetchone()
        if data[0] != "":
            userID = data[0]
            print(userID)
            cursor.execute(" update db_flybear.t_clinic set entecode=%s where clinicid=%s ",(enteCode,userID))
            db.commit()
    except:
        db.rollback()
        import sys
        data = sys.exc_info()
        print("importEnteCode:{}".format(data))
    finally:
        pass




def compressDB(db,oldcursor,dbName,connNew):
    "压缩库中的所有表"
    try:
        print("compressDB  begin")
        with connNew.cursor(SSDictCursor) as cursor:
            cursor.execute(" select table_name from information_schema.tables "
                           "where table_schema=%s and row_format<>'Compressed' and engine='InnoDB' " ,(dbName))
            while 1:
                row = cursor.fetchone()
                if not row or row is None:
                    break
                for k,v in row.items():
                    if k.upper() == "TABLE_NAME":
                        print("压缩表 {}.{}".format(dbName,v))
                        oldcursor.execute("alter table {dbname}.{tablename} ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;".format(dbname=dbName,tablename=v))
                        db.commit()
            connNew.commit()
        print("compressDB  end")
    except:
        import sys
        data = sys.exc_info()
        print("compressDB:{}".format(data))


def compressTable(db,cursor,dbName,tableName):
    "压缩表"
    #避免SQL注入 参数化查询
    cursor.execute("select count(*) as count from information_schema.tables where table_schema=%s and table_name=%s " , (dbName, tableName))
    data = cursor.fetchone()
    if data[0] == 0:
        print("表不存在")
    else:
        print("压缩表 %s.%s" % (dbName, tableName))
        try:
            compressSql = " alter table {dbname}.{tablename} ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;".format(dbname=dbName,tablename=tableName)
            cursor.execute(compressSql)
            db.commit()
        except py.ProgrammingError as err:
            print("compressTable::ProgrammingError:{}".format(err))
            db.rollback()
        except py.IntegrityError as err:
            print("compressTable::IntegrityError:{}".format(err))
            db.rollback()
        except py.DatabaseError as err:
            print("compressTable::DatabaseError:{errinfo}".format(errinfo=err))
            db.rollback()
        except py.MySQLError as err:
            print("compressTable::MySQLError:{errinfo}".format(errinfo=err))
            db.rollback()
        except py.Error as err:
            print("compressTable::Error:{}".format(err))
            db.rollback()
        except:
            import sys
            data = sys.exc_info()
            print("compressTable:%s"%(data))
            db.rollback()


def AddIndex(db,cursor,dbName,tableName):
    if dbName == "db_image" and tableName == "t_image":
        for i in range(3,200):
            theName = tableName + '_' + str(i)
            try:
                print(theName)
                cursor.execute(" alter table %s.%s add index ishandle(`ishandle`);alter table %s.%s add index classes(`classes`);alter table %s.%s add index ismanual(`ismanual`); alter table %s.%s add index aiclasses(`aiclasses`); " % (dbName, theName,dbName, theName,dbName, theName,dbName, theName))
                db.commit()
            except:
                import sys
                data = sys.exc_info()
                print(data)
                db.rollback()

            pass


def UpdateHandle(db,cursor,dbName,tableName):
    if dbName == "db_image" and tableName == "t_image":
        for i in range(200):
            theName = tableName + '_' + str(i)
            try:
                print(theName)
                cursor.execute(" update %s.%s  set ishandle=-1,aiclasses='',classes=if(ismanual=1,classes,'') where ishandle=0 or ishandle=-1 " % (dbName, theName))
                db.commit()
            except:
                import sys
                data = sys.exc_info()
                print(data)
                db.rollback()

def AddColumn(db,cursor,dbName,tableName,columnName,other):
    if dbName == "db_image" and tableName == "t_image":
        for i in range(200):
            theName = tableName+'_'+str(i)
            if HasColumn(cursor, dbName, theName, columnName) == False:
                print("添加字段 %s->%s" % (dbName + "." + theName, columnName))
                try:
                    cursor.execute(" ALTER TABLE %s.%s  ADD COLUMN %s %s " % (dbName, theName, columnName, other))
                    db.commit()
                except:
                    import sys
                    data = sys.exc_info()
                    print(data)
                    db.rollback()
        pass
    else:
        "添加字段"
        if HasColumn(cursor,dbName,tableName,columnName) == False:
            print("添加字段 %s->%s" %(dbName+"."+tableName,columnName))
            try:
                cursor.execute(" ALTER TABLE %s.%s  ADD COLUMN %s %s " %(dbName,tableName,columnName,other))
                db.commit()
            except:
                import sys
                data = sys.exc_info()
                print(data)
                db.rollback()

#alter table 表名 modify column 字段名 类型;

def ModifyColumn(db,cursor,dbName,tableName,columnName,other):
    if dbName == "db_image" and tableName == "t_image":
        for i in range(200):
            theName = tableName+'_'+str(i)
            if HasColumn(cursor, dbName, theName, columnName) == True:
                print("修改字段 %s->%s" % (dbName + "." + theName, columnName))
                try:
                    cursor.execute(" ALTER TABLE %s.%s  modify COLUMN %s %s " % (dbName, theName, columnName, other))
                    db.commit()
                except:
                    import sys
                    data = sys.exc_info()
                    print(data)
                    db.rollback()
        pass
    else:
        "修改字段"
        if HasColumn(cursor,dbName,tableName,columnName) == True:
            print("修改字段 %s->%s" %(dbName+"."+tableName,columnName))
            try:
                cursor.execute(" ALTER TABLE %s.%s  modify COLUMN %s %s " %(dbName,tableName,columnName,other))
                db.commit()
            except:
                import sys
                data = sys.exc_info()
                print(data)
                db.rollback()

def CreateTable(db,cursor,dbName,tableName,createSql):
    "创建表"
    cursor.execute("select count(*) as count from information_schema.tables where table_schema='%s' and table_name='%s' " %(dbName,tableName))
    data = cursor.fetchone()
    if data[0] == 0:
        print("创建表 %s.%s" %(dbName,tableName))
        try:
            print(createSql)
            cursor.execute(createSql)
            db.commit()
        except py.ProgrammingError as err:
            print("ProgrammingError:{}".format(err))
            db.rollback()
        except py.IntegrityError as err:
            print("IntegrityError:{}".format(err))
            db.rollback()
        except py.DatabaseError as err:
            print("DatabaseError:{errinfo}".format(errinfo = err))
            db.rollback()
        except py.MySQLError as err:
            print("MySQLError:{errinfo}".format(errinfo=err))
            db.rollback()
        except py.Error as err:
            print("Error:{}".format(err))
            db.rollback()
        except:
            import sys
            data = sys.exc_info()
            print(data)
            db.rollback()


def CreatePlanSetTemplate():
    sql = " CREATE TABLE db_koala.`t_plansettemplate` (" \
        " `ClinicUniqueID` varchar(32) NOT NULL COLLATE 'utf8_unicode_ci'," \
        " `PlanSetIdentity` VARCHAR(20) NOT NULL COLLATE 'utf8_unicode_ci'," \
        " `ItemIdentity` VARCHAR(20) NOT NULL COLLATE 'utf8_unicode_ci' COMMENT '项目ID'," \
        " `ItemTime` VARCHAR(10) NOT NULL DEFAULT '0' COLLATE 'utf8_unicode_ci' COMMENT '标准治疗时长' ," \
        " `ItemTimeUom` VARCHAR(10) NOT NULL DEFAULT '分钟' COLLATE 'utf8_unicode_ci' COMMENT '标准时长单位' ," \
         " `RetVisitItems` VARCHAR(2000) NOT NULL DEFAULT '' COLLATE 'utf8_unicode_ci' COMMENT '回访计划内容' ," \
        " `ScheduleItems` VARCHAR(2000) NOT NULL DEFAULT '' COLLATE 'utf8_unicode_ci' COMMENT '预约计划内容' ," \
        " `RetVisitDays` VARCHAR(100) NOT NULL DEFAULT '' COLLATE 'utf8_unicode_ci' COMMENT '回访天数' ," \
        " `ScheduleDays` VARCHAR(100) NOT NULL DEFAULT '' COLLATE 'utf8_unicode_ci' COMMENT '预约天数' ," \
        " `PlanStates` TINYINT(4) NOT NULL DEFAULT '1' COMMENT '启用状态1启0停'," \
        " `DisplayOrder` SMALLINT(5) NOT NULL DEFAULT '0'," \
        " `DataSource` TINYINT(4) NULL DEFAULT '1'," \
        " `Datastatus` TINYINT(4) NULL DEFAULT '1' COMMENT '状态'," \
        " `UpdateTime` DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)," \
        " `LastOperator` VARCHAR(10) NOT NULL DEFAULT 'PC' COLLATE 'utf8_unicode_ci'," \
        " `PCUpdateTime` DATETIME NULL DEFAULT NULL COMMENT 'pc端最后修改时间'," \
        " PRIMARY KEY (`ClinicUniqueID`,`PlanSetIdentity`)," \
        " INDEX `PlanSetIdentity` (`PlanSetIdentity`)," \
        " INDEX `ItemIdentity` (`ItemIdentity`)," \
        " INDEX `DisplayOrder` (`DisplayOrder`)," \
        " INDEX `PlanStates` (`PlanStates`) " \
        ") " \
        "COLLATE='utf8_unicode_ci' " \
        " ENGINE = InnoDB " \
        " PARTITION BY KEY (ClinicUniqueID)" \
        " PARTITIONS 10 ;"
    return  sql





def CreateQcItemTime():
    sql = " CREATE TABLE db_koala.`t_qcitemtime` " \
    "( `ClinicUniqueID` varchar(32) NOT NULL COLLATE 'utf8_unicode_ci'," \
    "`itqcIentity` VARCHAR(20) NOT NULL COLLATE 'utf8_unicode_ci'," \
    "`itqcPatIdentity` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
    "`itqcuserIdentity` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
    "`itqcitemIdentity` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '超时原因'," \
    "`qcItemStudyIdentiy` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
    "`itqcitemName` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '项目名称'," \
    "`itqctime` INT(4) NULL DEFAULT NULL COMMENT '标准时长'," \
    "`itqctimeunit` VARCHAR(8) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '标准时长单位'," \
    "`itqcState` INT(2) NULL DEFAULT NULL COMMENT '是否超时'," \
    "`itqcOvertime` INT(4) NULL DEFAULT NULL COMMENT '超时时长'," \
    "`itqcOvertimeunit` VARCHAR(8) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '超时时长单位' ," \
    "`itqcstarttime` DATETIME NULL DEFAULT NULL COMMENT '项目开始时间'," \
    "`itqcendtime` DATETIME NULL DEFAULT NULL COMMENT '项目结束时间'," \
    "`itqcOverRemark` VARCHAR(500) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '超时原因' ," \
    " `Datastatus` TINYINT(4) NULL DEFAULT '1' COMMENT '状态'," \
    " `UpdateTime` DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)," \
    " `LastOperator` VARCHAR(10) NOT NULL DEFAULT 'PC' COLLATE 'utf8_unicode_ci'," \
    " `PCUpdateTime` DATETIME NULL DEFAULT NULL COMMENT 'pc端最后修改时间'" \
    ")" \
    "COLLATE='utf8_unicode_ci' " \
    "ENGINE=InnoDB " \
    " PARTITION BY KEY (ClinicUniqueID) " \
    " PARTITIONS 10 "
    return sql

def CreateQcItemSet():
    sql = "CREATE TABLE db_koala.`t_qcitemset` (" \
        " `ClinicUniqueID` varchar(32) NOT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcitemsetIdentity` VARCHAR(20) NOT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcitemIdentity` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '项目ID' ," \
        "`qcItemStudyIdentiy` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcitem` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '项目名称' ," \
        "`qcitstandertime` INT(4) NULL DEFAULT NULL COMMENT '项目标准时长'," \
        "`qcitstandertimeunit` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '时长单位' ," \
        "`warnUserGroupid` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '预警人' ," \
        "`warnUserGroup` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        " `Datastatus` TINYINT(4) NULL DEFAULT '1' COMMENT '状态'," \
        " `UpdateTime` DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)," \
        " `LastOperator` VARCHAR(10) NOT NULL DEFAULT 'PC' COLLATE 'utf8_unicode_ci'," \
        " `PCUpdateTime` DATETIME NULL DEFAULT NULL COMMENT 'pc端最后修改时间'," \
        "PRIMARY KEY (`ClinicUniqueID`,`qcitemsetIdentity`)," \
        "INDEX `qcitemsetIdentity` (`qcitemsetIdentity`)," \
        "INDEX `qcitemIdentity` (`qcitemIdentity`)," \
        "INDEX `warnUserGroupid` (`warnUserGroupid`)," \
        "INDEX `qcItemStudyIdentiy` (`qcItemStudyIdentiy`))" \
        "COLLATE='utf8_unicode_ci'" \
        "ENGINE=InnoDB " \
        " PARTITION BY KEY (ClinicUniqueID)" \
        " PARTITIONS 10 "
    return  sql

def CreateQcPlan():
    sql = "CREATE TABLE db_koala.`t_qcplan` (" \
        "`ClinicUniqueID` varchar(32) NOT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcplanIdentity` VARCHAR(20) NOT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcsetIdentity` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '对应的质控设置' ," \
        "`FirstItemIdentity` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`FirstItemName` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcItemType` INT(2) NULL DEFAULT NULL COMMENT '1:项目，2:操作'," \
        "`qcItemIdentity` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '质控项目ID qcitemtype为2时 ID 1:预约（预约确认时间），2回访（具体回访时间），3到诊（到诊时）' ," \
        "`qcItemName` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '质控项目名称' ," \
        "`qcItemStudyIdentiy` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '产生质控的就诊' ," \
        "`qcItemcompletestu` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '完成质控的就诊' ," \
        "`qcPatientId` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '患者' ," \
        "`qcplanTimeLen` INT(4) NULL DEFAULT NULL COMMENT '标准时长'," \
        "`qcplanTimeunit` VARCHAR(5) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '标准时长单位' ," \
        "`qcuserid` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '生成质控事件的用户' ," \
        "`qcDealUserid` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '处理质控事件的用户' ," \
        "`qcDathTime` DATETIME NULL DEFAULT NULL COMMENT '事件正常结束时间'," \
        "`qcendDateTime` DATETIME NULL DEFAULT NULL COMMENT '事件实际结束时间'," \
        "`qcBeginTime` DATETIME NULL DEFAULT NULL COMMENT '生成质控计划的时间 质控开始时间'," \
        "`qcwarnTime` DATETIME NULL DEFAULT NULL COMMENT '质控开始预警时间'," \
        "`qcPlanState` INT(2) NULL DEFAULT NULL COMMENT '0计划中未处理，1正常处理，2，超期未处理，3超期处理，4不处理'," \
        "`qcdealRemark` VARCHAR(400) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '处理或不处理的备注' ," \
        "`qcEndTimelen` INT(4) NULL DEFAULT NULL COMMENT '超期时间（或还剩多长时间）+-标示'," \
        "`qcEndTimeUnit` VARCHAR(5) NULL DEFAULT NULL  COLLATE 'utf8_unicode_ci' COMMENT '超期时间单位'," \
        "`qcdeal` INT(2) NULL DEFAULT NULL COMMENT '是否已处理'," \
        "`message` VARCHAR(400) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`Datastatus` TINYINT(4) NULL DEFAULT '1' COMMENT '状态'," \
        "`UpdateTime` DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)," \
        "`LastOperator` VARCHAR(10) NOT NULL DEFAULT 'PC' COLLATE 'utf8_unicode_ci'," \
        "`PCUpdateTime` DATETIME NULL DEFAULT NULL COMMENT 'pc端最后修改时间'," \
        "PRIMARY KEY (`ClinicUniqueID`,`qcplanIdentity`)," \
        "INDEX `qcplanIdentity` (`qcplanIdentity`)," \
        "INDEX `qcsetIdentity` (`qcsetIdentity`)," \
        "INDEX `qcItemIdentity` (`qcItemIdentity`)," \
        "INDEX `qcPatientId` (`qcPatientId`)," \
        "INDEX `qcuserid` (`qcuserid`)," \
        "INDEX `qcDealUserid` (`qcDealUserid`)," \
        "INDEX `qcdeal` (`qcdeal`)," \
        "INDEX `qcItemStudyIdentiy` (`qcItemStudyIdentiy`)," \
        "INDEX `FirstItemIdentity` (`FirstItemIdentity`)," \
        "INDEX `qcItemcompletestu` (`qcItemcompletestu`))" \
        "COMMENT='质控计划表'" \
        "COLLATE='utf8_unicode_ci'" \
        "ENGINE=InnoDB " \
        "PARTITION BY KEY (ClinicUniqueID)" \
        "PARTITIONS 10 "
    return sql

def CreateQcSet():
    sql = "CREATE TABLE db_koala.`t_qcset` (" \
        "`ClinicUniqueID` varchar(32) NOT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcIdentity` VARCHAR(20) NOT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcFirstItem` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcFirstItemName` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcFirstType` INT(2) NULL DEFAULT NULL," \
        "`qcTimeLen` INT(6) NULL DEFAULT NULL," \
        "`qcTimeUnit` VARCHAR(2) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcwarntime` INT(2) NULL DEFAULT NULL," \
        "`qcwarntimeUnit` VARCHAR(2) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcSecondItem` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '质控项目ID qcitemtype为2时 ID 1:预约（预约确认时间），2回访（具体回访时间），3到诊（到诊时间）' ," \
        "`qcSeconItemName` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcSecondType` INT(2) NULL DEFAULT NULL COMMENT '1:项目，2:操作'," \
        "`qcUserGroupId` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcUserGroup` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcUserId` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcUser` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcContent` VARCHAR(2000) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci'," \
        "`qcState` VARCHAR(20) NULL DEFAULT NULL COLLATE 'utf8_unicode_ci' COMMENT '0停用1启用' ," \
        "`Datastatus` TINYINT(4) NULL DEFAULT '1' COMMENT '状态'," \
        "`UpdateTime` DATETIME(6) NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)," \
        "`LastOperator` VARCHAR(10) NOT NULL DEFAULT 'PC' COLLATE 'utf8_unicode_ci'," \
        "`PCUpdateTime` DATETIME NULL DEFAULT NULL COMMENT 'pc端最后修改时间'," \
        "PRIMARY KEY (`ClinicUniqueID`,`qcIdentity`)," \
        "INDEX `qcIdentity` (`qcIdentity`)," \
        "INDEX `qcUserGroupId` (`qcUserGroupId`)" \
        ")" \
        "COLLATE='utf8_unicode_ci'" \
        "ENGINE=InnoDB " \
        "PARTITION BY KEY (ClinicUniqueID)" \
        "PARTITIONS 10 "
    return sql
