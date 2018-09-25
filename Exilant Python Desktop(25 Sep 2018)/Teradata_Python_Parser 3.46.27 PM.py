
import re
import os, subprocess, tempfile
import sys
import csv
import json
import shutil
import logging
import datetime
from subprocess import call
#############################################################################################################################################        
## Funtion Name: ensure_dir
## Functionality: check if the directory exists
## Input: a path as String
## Output:
#############################################################################################
def ensure_dir(dir_path):
    #directory = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
	
    ##else:
    ##    shutil.rmtree(dir_path)
    ##    os.makedirs(dir_path)
#############################################################################################################################################        
## Funtion Name: exist_func
## Functionality: check if element of 1st list exists in the 2nd list. 
##                If exists then returns the index of the element with in 1st list.
## Input: 2 lists
## Output: index as integer
#############################################################################################
def exist_func(a, b):
    match_index = -1
    for i in a:
       if str(i).upper() in b:
          match_index = a.index(i)
          break
    return match_index
#############################################################################################################################################        
## Funtion Name: table_syntax_checker
## Functionality: check if the table definition is executable in database by using bteq block. 
## Input: table definition
## Output: True/False
#############################################################################################
def table_syntax_checker(table_syntax,fileDir,log_dir_path):
    ##table_syntax = re.sub('\s+',' ',table_syntax.replace('\n',' ').replace('.',' .'))
    table_syntax = table_syntax.replace('.',' .') #.replace('\n',' ')
    table_syntax = re.sub('(?i)TABLE\s+[a-zA-Z0-9_]+\s+.', 'TABLE DM_APP.',table_syntax)
    table_syntax = table_syntax.replace(' .','.')
    pattern = re.compile("COMMENT ON",re.IGNORECASE)
    for m in re.finditer(pattern, table_syntax):
    	if(m.start(0) > 0):
    		table_syntax = table_syntax[:m.start(0) - 1]
    		break
			
    table_syntax = "Explain " + table_syntax + ";"
    table_syntax = re.sub(';+',';',table_syntax)
    expln_temp_file = "explain_file"
    expln_temp_file = os.path.join(log_dir_path, expln_temp_file)
    
    with open(expln_temp_file, 'w') as file:
         file.write(table_syntax)
         
    command = "sh " + fileDir + "/bteq_script.ksh " + expln_temp_file + " " + log_dir_path
    os.system(command)
    
    rslt_file = "expain_chk.txt"
    
    errorCode = 0
    with open(os.path.join(log_dir_path, rslt_file),mode='r') as fi:
	    errorCode = int(fi.read())
    
    #return True
    if(errorCode==0):
        return True
    else:
        return False
	
#############################################################################################################################################        
## Funtion Name: table_tokenizer
## Functionality: Scan each line file content and find out tablename 
##                and corresponding table properties,column and comment info
## Input: a directory path where frameowrk specific imtermediate temp table structure get stored,
##        a list of dictionaries which contains filename and it's content,
##        a dictionary of classwords,datatype and allowable data length(metadata info)
## Output: a dictionary of tables  with corresponding table properties,column and comment info  
###################################################################################################
        
def table_tokenizer(dlldir,filesddllst,coldttpszdict,log_dir_path,fileDir):
    coldatatypedictlist = coldttpszdict.values()
       
    datatypedict = {}
       
    for item in coldatatypedictlist:
        datatypedict.update(item)
    
    datatypelist = set(datatypedict.keys())
    ######print datatypelist
    
    table_word_map_dict = {}
    print range(len(filesddllst))    
    for i in range(len(filesddllst)):
       filename = filesddllst[i]["fileName"]
       filedata = filesddllst[i]["content"].replace("\n"," ").replace("\\n"," ").replace('<dm tool line brk is here>','\n')  ##.replace("\\","")
       syntax_chk = table_syntax_checker(filedata,fileDir,log_dir_path)
       #print "syntax_chk is:"
       #print syntax_chk
       #print "======================= FILE DATA =========================="
       #print filedata
	   
       # Write filedata to file for processing of COMPRESS
       modfilepath = dlldir + "/" + filename.split("/")[len(filename.split("/")) - 1]
       with open(modfilepath, 'w') as file:
          file.write(filedata)
       raw_line = []	  
       tmp_file = ""
       with open(modfilepath,'r') as fi:
          for line in fi:
              line = re.sub('\s+', ' ',line).replace(' .', '.').replace('. ', '.').replace(' =', '=').replace('= ', '=') 			  
              raw_line.append(line)
       #Handle commas inside COMPRESS by removing it
       for li in raw_line: 
          if ("COMPRESS" in li):		  
              strtidx =  li.index("COMPRESS")
              new_line = li[:strtidx] + li[strtidx: -2].replace(',',':').strip() + li.strip()[-1]  
              new_line = new_line.replace("':'", "','")			  
              print "============== NEW LINE ==================="
              print new_line			  
          else:
              new_line = li 		  
          tmp_file = tmp_file + new_line + "\n"
       filedata = tmp_file
       #print "================ NEW FILE DATA =====================" 
       #print filedata	  
	   
       if(syntax_chk):
           # Replace the target string
          modfilepath = dlldir + "/" + filename.split("/")[len(filename.split("/")) - 1]
          filedata = re.sub('\s+',' ',filedata.replace('\n',' '))
          filedata = filedata.replace('(', ' ( ').replace(')', ' ) ').replace("'", " ' ").replace(',', ' ,\n').replace(";", " ;\n")##.replace("COMMENT ON","\n COMMENT ON").replace("PARTITION BY","\n PARTITION BY").replace("PRIMARY INDEX","\n PRIMARY INDEX").replace("NO FALLBACK","\n NO FALLBACK").replace("NO BEFORE JOURNAL","\n NO BEFORE JOURNAL").replace("NO AFTER JOURNAL","\n  NO AFTER JOURNAL").replace("CHECKSUM DEFAULT","\n  CHECKSUM DEFAULT").replace("DEFAULT MERGEBLOCKRATIO","\n  DEFAULT MERGEBLOCKRATIO")
          pattern = re.compile("COMMENT ON",re.IGNORECASE)
          filedata = re.sub(pattern,'\n COMMENT ON',filedata)
          pattern = re.compile("PARTITION BY",re.IGNORECASE)
          filedata = re.sub(pattern,'\n PARTITION BY',filedata)
          pattern = re.compile("PRIMARY INDEX",re.IGNORECASE)
          filedata = re.sub(pattern,'\n PRIMARY INDEX',filedata)
          pattern = re.compile("NO FALLBACK",re.IGNORECASE)
          filedata = re.sub(pattern,'\n NO FALLBACK',filedata)
          pattern = re.compile("NO BEFORE JOURNAL",re.IGNORECASE)
          filedata = re.sub(pattern,'\n NO BEFORE JOURNAL',filedata)
          pattern = re.compile("NO AFTER JOURNAL",re.IGNORECASE)
          filedata = re.sub(pattern,'\n  NO AFTER JOURNAL',filedata)
          pattern = re.compile("CHECKSUM DEFAULT",re.IGNORECASE)
          filedata = re.sub(pattern,'\n  CHECKSUM DEFAULT',filedata)
          pattern = re.compile("DEFAULT MERGEBLOCKRATIO",re.IGNORECASE)
          filedata = re.sub(pattern,'\n  DEFAULT MERGEBLOCKRATIO',filedata)
          pattern = re.compile("INEDX",re.IGNORECASE)
          filedata = re.sub(pattern,'INDEX',filedata)
          pattern = re.compile("TITLE",re.IGNORECASE)
          filedata = re.sub(pattern,'TITLE',filedata)
          pattern = re.compile("COMPRESS",re.IGNORECASE)
          filedata = re.sub(pattern,'COMPRESS',filedata)
          pattern = re.compile("AS",re.IGNORECASE)
          filedata = re.sub(pattern,'AS',filedata)
          pattern = re.compile("ALWAYS",re.IGNORECASE)
          filedata = re.sub(pattern,'ALWAYS',filedata)
          pattern = re.compile("GENERATED",re.IGNORECASE)
          filedata = re.sub(pattern,'GENERATED',filedata)
          pattern = re.compile("TABLE",re.IGNORECASE)
          filedata = re.sub(pattern,'TABLE',filedata)
          pattern = re.compile("MULTISET",re.IGNORECASE)
          filedata = re.sub(pattern,'MULTISET',filedata)
		  
          #print "===============File Data=============="
          #print filedata
		  
          # Write the file out again
          with open(modfilepath, 'w') as file:
            file.write(filedata)
       
          with open(modfilepath,'r') as fi:
              line_cnt = 0
              tablename = ""
              table_properties_dict = {}
              column_properties_dict = {}
              set_properties = ""
              jarnal_bfor_flag = "N"
              jarnal_aftr_flag = "N"
              fallback_flag = "N"
              chksm_flag = "N"
              blk_rto_flag = "N"
              table_snap_properties = "N"
              primary_index=""
              pi_flag = False
              comment_list = []
              column_start_flag = False
              column_end_flag = False
              partition_flag = False
              title_flag=False
              colsizeflag=False
              tabl_lvl_identity_col_flag = "N"
              column_continuation_flag = "N"
              sq_continuation_flag = "N"
              partition_cols = []
              dbname = ""
              size=""
              for line in fi:
                  line = re.sub('\s+', ' ',line).replace(' .', '.').replace('. ', '.').replace(' =', '=').replace('= ', '=')
                  srchsplit = re.split('[^a-zA-Z0-9,._)(;"\'$]',line.strip())
                  
				  #print "============ SRCHSPLIT =========="				  
                  #print srchsplit
                  datatypeExistsIndex = -1
                          
                  if(len(srchsplit) > 1):
                  
                      if(not column_start_flag):
                          line_cnt = line_cnt + 1
                          if((len(srchsplit) > 2) and ("TABLE" in srchsplit)):
                              tablename = srchsplit[srchsplit.index('TABLE') + 1]
                              if(len(tablename.split(".")) > 1):
                                dbname = tablename.split(".")[0]
                              prev_wrd_of_tbl = srchsplit[srchsplit.index('TABLE') - 1]
                              
                              if(prev_wrd_of_tbl != "MULTISET"):
                                  set_properties = "SET"
                              else:
                                  set_properties = "MULTISET"
                                  
                          if ((len(srchsplit) > 2) and ("JOURNAL" in srchsplit)): 
                              if((srchsplit[srchsplit.index('JOURNAL') - 1] == "BEFORE") 
                                 and (srchsplit[srchsplit.index('JOURNAL') - 2] == "NO")):
                                  jarnal_bfor_flag = "Y"
                              
                              if((srchsplit[srchsplit.index('JOURNAL') - 1] == "AFTER") 
                                 and (srchsplit[srchsplit.index('JOURNAL') - 2] == "NO")):
                                  jarnal_aftr_flag = "Y"
                          
                              
                          if ((len(srchsplit) > 1) and ("FALLBACK" in srchsplit)):
                              if(srchsplit[srchsplit.index('FALLBACK') - 1] == "NO"):
                                  fallback_flag = "Y"
                              else:
                                  fallback_flag = "N"
                              
                          if ((len(srchsplit) > 1) and ("CHECKSUM" in srchsplit)):
                              if(srchsplit[srchsplit.index('CHECKSUM') + 1] == "DEFAULT"):
                                  chksm_flag = "Y"
                              else:
                                  chksm_flag = "N"
                          
                          if ((len(srchsplit) > 1) and ("MERGEBLOCKRATIO" in srchsplit)):
                              if(srchsplit[srchsplit.index('MERGEBLOCKRATIO') - 1] == "DEFAULT"):
                                  blk_rto_flag = "Y"
                              else:
                                  blk_rto_flag = "N"
								  
                      #print("column_continuation_flag is : %s" %column_continuation_flag)
                      #print("sq_continuation_flag is : %s" %sq_continuation_flag)
 
                      if(("INDEX" in srchsplit) and (srchsplit[srchsplit.index('INDEX') - 1] == "PRIMARY")
                         and (((column_continuation_flag == "N") and (sq_continuation_flag == "N")) and (datatypeExistsIndex < 0))):
                          pi_flag = True
                          primary_index == ""
                         
                      if(pi_flag):
                          if(primary_index == ""):
                            if("(" in srchsplit[srchsplit.index("INDEX"):]):
                                primary_index = srchsplit[srchsplit.index("(") + 1]
                            else:
                                pi_flag = False
                            if(pi_flag):
                                if(primary_index.startswith('"') and not primary_index.endswith('"')):
                                   cnt = 1
                                   loop_start_index = srchsplit.index("(") + cnt
                                   while(loop_start_index < len(srchsplit)):
                                      loop_start_index = loop_start_index + cnt
                                      if(srchsplit[loop_start_index].endswith('"')):
                                         primary_index = primary_index + " " + srchsplit[loop_start_index]
                                         break
                                      else:
                                         primary_index = primary_index + " " + srchsplit[loop_start_index]
                                if(")" in srchsplit):
                                   pi_flag = False         
                          elif(")" not in srchsplit):
                            primary_index = primary_index + "," + srchsplit[0]
                            if(srchsplit[0].startswith('"') and not srchsplit[0].endswith('"')):
                                cnt = 1
                                loop_start_index = 0
                                while(loop_start_index < len(srchsplit)):
                                   loop_start_index = loop_start_index + cnt
                                   ######print srchsplit[loop_start_index]
                                   if(srchsplit[loop_start_index].endswith('"')):
                                      primary_index = primary_index + " " + srchsplit[loop_start_index]
                                      break
                                   else:
                                      primary_index = primary_index + " " + srchsplit[loop_start_index]
                          else:
                            temp_pi = srchsplit[srchsplit.index(")") - 1]
                            if(srchsplit[srchsplit.index(")") - 1].endswith('"') and not srchsplit[srchsplit.index(")") - 1].startswith('"')):
                                last_pi_index = srchsplit.index(")") - 1
                                cnt = 0
                                while((last_pi_index - cnt ) >= 1 ):
                                   cnt = cnt + 1
                                   if(srchsplit[last_pi_index - cnt].startswith('"')):
                                      temp_pi = srchsplit[last_pi_index - cnt] + " " + temp_pi
                                      break
                                   else:
                                      temp_pi = srchsplit[last_pi_index - cnt] + " " + temp_pi
                                      
                            primary_index = primary_index + "," + temp_pi
                            pi_flag = False
                            
                      if((len(srchsplit) >= 2) and ("PARTITION" in srchsplit) and (srchsplit[srchsplit.index('PARTITION') + 1] == "BY")
                         and (((column_continuation_flag == "N") and (sq_continuation_flag == "N")) and (datatypeExistsIndex < 0))):      
                          column_end_flag = True
                          partition_flag = True
                          
                      if(partition_flag):
                          if(("INDEX" in srchsplit) or (";" in srchsplit)):
                            if("INDEX" in srchsplit): 
                                partition_cols = partition_cols + srchsplit[:srchsplit.index("INDEX") -1]
                            else:
                                partition_cols = partition_cols + srchsplit[:srchsplit.index(";") -1]
                            partition_flag = False
                          else:
                            partition_cols = partition_cols + srchsplit    
                      
                           
                      datatypeExistsIndex = exist_func(srchsplit,datatypelist)
                      if ((datatypeExistsIndex >= 0 or column_continuation_flag != "N") and not column_end_flag):
                          if(colsizeflag):
                            if(len(srchsplit)>1 and srchsplit[0].isdigit()):
                               size = size + "," + srchsplit[0]
                               colsizeflag = False
                          ##print size      
                          if(datatypeExistsIndex > 0 and column_continuation_flag == "N"
                             and ((not(line.strip().startswith("'"))) and (sq_continuation_flag == "N"))):  ##if column_continuation_flag != N then column chk won't happen
                              column_start_flag = True
                              identity_col_flag = "N"
                              columnname = ""
                              coldatatype = ""
                              commentlst = []
                              size = ""
                              title = ""
                              compress_flag = "N"
                              sq_srch_strt_pos = 0
                              indexofmtchwrd =  datatypeExistsIndex ##columnsrchsplit.index(datatypeword)
                              indexofcolumn = indexofmtchwrd - 1
                              columnname = srchsplit[indexofcolumn]
                              size=""
                              colsizeflag = False
                              ######print columnname
                              if(columnname.endswith('"') and not columnname.startswith('"')):
                                cnt = 0
                                while((indexofcolumn - cnt ) >= 1 ):
                                   cnt = cnt + 1
                                   if(srchsplit[indexofcolumn - cnt].startswith('"')):
                                      columnname = srchsplit[indexofcolumn - cnt] + " " + columnname
                                      break
                                   else:
                                      columnname = srchsplit[indexofcolumn - cnt] + " " + columnname
                              coldatatype = srchsplit[indexofmtchwrd]
                              
                                        
                              if(len(srchsplit) >= 5):
                                 indexoffrstbrckt = indexofmtchwrd + 1
                                 indexofSZdata = indexofmtchwrd + 2
                                 indexofnxttosz = indexofmtchwrd + 3
                                 
                                 if(srchsplit[indexoffrstbrckt] == "("):
                                      if(srchsplit[indexofSZdata].isdigit()):
                                          size = srchsplit[indexofSZdata].split(',')[0]
                                          if(srchsplit[indexofnxttosz] == ","):
                                            colsizeflag = True
                          
                                        
                          if(("TITLE" in srchsplit) and (srchsplit[srchsplit.index("TITLE") + 1]=="'")):
                             a = re.search(r'\b((?i)TITLE)\b',line.strip())
                             srch_strt_idx=a.start() + 8
                             ##print line.strip()[srch_strt_idx:]
                             title_flag=True
                             for x in line.strip()[srch_strt_idx:]:
                               if(x == "'"):
                                  title_flag=False
                                  break
                               title = title + x
                          if(title_flag) and (column_continuation_flag == "SQ"):
                            for x in line.strip():
                                if(x == "'"):
                                  title_flag=False
                                  break
                                title = title + x   
                                               
                          if("COMPRESS" in srchsplit):
                             a = re.search(r'\b((?i)COMPRESS)\b',line.strip())
                             if(a.start() > 0):
                                compress_flag = "Y"
                                            
                          if("IDENTITY" in srchsplit[(indexofmtchwrd + 1):]):
                             index_of_as = srchsplit.index("IDENTITY") - 1
                             index_of_always = srchsplit.index("IDENTITY") - 2
                             index_of_genratd = srchsplit.index("IDENTITY") - 3
                             
                             ####print srchsplit
                             
                             if(srchsplit[index_of_as] == "AS" and srchsplit[index_of_always] == "ALWAYS" 
                                and srchsplit[index_of_genratd] == "GENERATED"):
                                    identity_col_flag = "Y"
                                    tabl_lvl_identity_col_flag = "Y"
                                    
                                    
                          if(")" in srchsplit and column_continuation_flag == "OB"):
                             column_continuation_flag = "N"
                          elif("'" in srchsplit and column_continuation_flag == "SQ"):
                             column_continuation_flag = "N"
                             sq_srch_strt_pos = srchsplit.index("'") + 1
                  
                          if(("(" in srchsplit and ")" in srchsplit and srchsplit.count("(") > srchsplit.count(")")
                              and len(srchsplit) - 1 - srchsplit[::-1].index("(") > len(srchsplit) - 1 - srchsplit[::-1].index(")")) 
                             or ("(" in srchsplit[indexofcolumn + 1:] and ")" not in srchsplit[indexofcolumn + 1:])):
                             ##column_start_line_cnt = line_cnt
                             column_continuation_flag = "OB"
                          
                          mod_srchsplit_list = [item.split('.') for item in srchsplit[sq_srch_strt_pos:]]
                          mod_srchsplit = []
						  
                          for item in mod_srchsplit_list:
                             mod_srchsplit = mod_srchsplit + item
                          
                          if(("'" in srchsplit) and (mod_srchsplit.count("'") % 2 != 0)):
                             column_continuation_flag = "SQ"
                             ##print srchsplit
                          ##print ("size here is " + size + column_continuation_flag)   
                          if(column_continuation_flag == "N"):
                            ##print ("size here is " + size + column_continuation_flag)
                            column_properties_dict.update({columnname:{"col_data_type":coldatatype.upper(),
                                                                       "col_size":size,
                                                                       "identity_flag":identity_col_flag,
                                                                       "title":title.strip(),
                                                                       "compress_flag":compress_flag}})
																	   
                      if(line.strip().replace(","," ").strip().replace(") )","))").strip(")").strip().endswith("'") and (line.strip().count("'") % 2 != 0)
                         and sq_continuation_flag == "SQ"):
                           sq_continuation_flag = "N"
                           column_continuation_flag = "N"
                                               
                      if((line.strip().startswith("'")) and (line.strip().count("'") % 2 != 0)):
                           sq_continuation_flag = "SQ"
                      if(("COMMENT" in srchsplit) and (srchsplit[srchsplit.index("COMMENT") + 1] == "ON")):
                           comment_list.append(srchsplit)
              if(jarnal_bfor_flag == jarnal_aftr_flag == fallback_flag == chksm_flag == blk_rto_flag == "Y"):
                    table_snap_properties = "Y" 
              if(tablename != ""):                                       
                table_word_map_dict.update({tablename:{"SYNTAX_CHK":"Pass",
                                                      "DBNAME":dbname,
                                                      "SET_PROPERTIES":set_properties,
                                                      "TABLE_SNAP_PROPERTIES":table_snap_properties,
                                                      "PRIMARY_INDEX":primary_index,
                                                      "IDENTITY_FLAG":tabl_lvl_identity_col_flag,
                                                      "COLUMN_DESC":column_properties_dict,
                                                      "COMMENTS_LIST":comment_list,
                                                      "PARTITION_COLS":partition_cols,
                                                      "FILENAME":filename}})
                                                    
       else:
            tablename = "UNKNOWN"
            dbname = ""
            modfilepath = dlldir + "/" + filename.split("/")[len(filename.split("/")) - 1]
           
            filedata = re.sub('\s+',' ',filedata.replace('\n',' '))
            filedata = filedata.upper().replace('(', ' ( ').replace(')', ' ) ').replace("'", " ' ").replace(',', ' ,\n').replace(";", " ;\n").replace("COMMENT ON","\n COMMENT ON").replace("PARTITIONED BY","\n PARTITIONED BY").replace("PRIMARY INDEX","\n PRIMARY INDEX").replace("NO FALLBACK","\n NO FALLBACK").replace("NO BEFORE JOURNAL","\n NO BEFORE JOURNAL").replace("NO AFTER JOURNAL","\n  NO AFTER JOURNAL").replace("CHECKSUM DEFAULT","\n  CHECKSUM DEFAULT").replace("DEFAULT MERGEBLOCKRATIO","\n  DEFAULT MERGEBLOCKRATIO")
            
            # Write the file out again
            with open(modfilepath, 'w') as file:
              file.write(filedata)
              
            with open(modfilepath,'r') as fi:
                for line in fi:
                    line = re.sub('\s+', ' ',line).replace(' .', '.').replace('. ', '.')
                    
                    srchsplit = re.split('[^a-zA-Z0-9,._)("]',line.upper().strip())
                    
                    if((len(srchsplit) > 2) and ("TABLE" in srchsplit)):
                          tablename = srchsplit[srchsplit.index('TABLE') + 1]
                          if(len(tablename.split(".")) > 1):
                                dbname = tablename.split(".")[0]
                          
                table_word_map_dict.update({tablename:{"SYNTAX_CHK":"Fail",
                                                       "DBNAME":dbname,
                                                       "FILENAME":filename}})                
    print  table_word_map_dict          
    return table_word_map_dict
#############################################################################################################################################            
## Funtion Name: comment_PII_match
## Functionality: check if a column is having any comment section, if yes
##                then checks if valid PII word exists in the comment in the format like
##				  COMMENT ON COLUMN <db_name>.<tablename>.<column_name> IS 'PII_[PII Code]:[optional column comments or description]'
## Input: output dictionary of table_tokenizer function, 
##	      a list of valid PII words(metadata info)
## Output: dictionay of tables and corresponding columns which violates the scenarios mentioned above
#######################################################################################################
def comment_PII_match(table_col_comment_dict,PII_list):
    keylist = table_col_comment_dict.keys()
    
    ##x = 5/0
    
    table_col_PII_desc_dict = {}
    
    for ky in keylist:
        tablename = ky
        keyval = table_col_comment_dict.get(ky)
        syntax_chk_flag = keyval["SYNTAX_CHK"]
        if(syntax_chk_flag == "Pass"):
            columnlist = keyval["COLUMN_DESC"].keys()
            comment_list = keyval["COMMENTS_LIST"]
            refined_comment_list = []
            for each_list in comment_list:
              for item in each_list:
                if(item.endswith('"') and not item.startswith('"')):
                    indxofitem = each_list.index(item)
                    cnt = 0
                    while((indxofitem - cnt ) >= 1 ):
                       cnt = cnt + 1
                       if('"' in each_list[indxofitem - cnt]):
                          item = each_list[indxofitem - cnt] + " " + item
                          break
                       else:
                          item = each_list[indxofitem - cnt] + " " + item
                   
                temp_comment = item.upper().split(".")
                refined_comment_list = refined_comment_list + temp_comment
              refined_comment_list = refined_comment_list + [";"]
            print columnlist
            print refined_comment_list
            piiexcpcoldict = {} 
            
            for column in columnlist:
                pii_text = ""
                comment_flag = False
                if(column.upper() in refined_comment_list):
                    col_indx_in_cmnt = refined_comment_list.index(column.upper())
                    tmp_table = refined_comment_list[col_indx_in_cmnt - 1]
                    if(tmp_table.upper() in tablename.upper().split(".")):
                        comment_flag = True
                        semicol_indx_aftr_col = refined_comment_list.index(";",col_indx_in_cmnt + 1)
                        for pii_elmnt in PII_list:
                            if(pii_elmnt.upper() in refined_comment_list[(col_indx_in_cmnt + 1):semicol_indx_aftr_col]):
                                pii_text = pii_elmnt
                
                if(comment_flag):
                    if(pii_text != ""):
                        piiexcpcoldict.update({column:pii_text})
                    else:
                        piiexcpcoldict.update({column:"NO_PII"})
                else:
                    piiexcpcoldict.update({column:"NO_COMMENT"})
                
            table_col_PII_desc_dict.update({tablename:piiexcpcoldict})                           
        
    return  table_col_PII_desc_dict
#############################################################################################################################################            
## Funtion Name: class_word_match
## Functionality: check if the class word exist in a column name of a table 
## Input: output dictionary of table_tokenizer function,
##        a list of classwords(matadata info)
## Output: dictionay of tables and corresponding column names and classword where class word exists
####################################################################################################
def class_word_match(table_col_dict,clsword_lst):
    keylist = table_col_dict.keys()
    
    table_col_excp_dict = {}
    
    for ky in keylist:
        tablename = ky
        keyval = table_col_dict.get(ky)
        syntax_chk_flag = keyval["SYNTAX_CHK"]
        if(syntax_chk_flag == "Pass"):
            columnlist = keyval["COLUMN_DESC"].keys()
        
            claawrd_match_dict = {}
        
            for column in columnlist:
                clswrd_chk = False
                for clswrd in clsword_lst:
                    refined_column = re.sub(r'^"|"$', '', column)
                    if(refined_column.upper().endswith(clswrd.upper())):
                        clswrd_chk = True
                        break 
                if(clswrd_chk):
                    claawrd_match_dict.update({column:clswrd})
        
            table_col_excp_dict.update({tablename:claawrd_match_dict})
    
    return table_col_excp_dict
#############################################################################################################################################            
## Funtion Name: naming_standard_match
## Functionality: check if a column is following the naming standard
##                as mentioned in https://connectme.apple.com/docs/DOC-925643
## Input: output dictionary of table_tokenizer function,
##		  a dictionary which contains actual word and the abbreviation that should be used(metadata info)
## Output: dictionay of tables and corresponding faulty columns and abbreviation
############################################################################################# 
def naming_standard_match(table_col_dict,naming_cnv_dict):
    keylist = table_col_dict.keys()
    namingsrchllstkeys = naming_cnv_dict.keys()
    namingsrchllstvalues = naming_cnv_dict.values()
    
    table_col_excp_dict = {}
    
    for ky in keylist:
        tablename = ky
        keyval = table_col_dict.get(ky)
        syntax_chk_flag = keyval["SYNTAX_CHK"]
        if(syntax_chk_flag == "Pass"):
            columnlist = keyval["COLUMN_DESC"].keys()
        ######print tablename
        ######print columnlist
        
            namingexcptn_col_dict = {}
            
            for column in columnlist:
                col_wrd_lst = re.split('[^a-zA-Z0-9]', column.upper().strip())
                conv_flag=True
                for wrd in col_wrd_lst:
                    if(wrd in namingsrchllstkeys):
                        tobeused = naming_cnv_dict[wrd]
                        namingexcptn_col_dict.update({column:"Instead of '" + wrd + "' '" + tobeused + "' should be used"})
                    elif(wrd in namingsrchllstvalues):
                        conv_flag=True
                    else:
                        conv_flag=False
            table_col_excp_dict.update({tablename:namingexcptn_col_dict})
    
    return  table_col_excp_dict 
        
#############################################################################################################################################            
## Funtion Name: reserved_word_chk
## Functionality: check if a column name refers to a Reserved word in Hive/Hadoop
## Input: output dictionary of table_tokenizer function,
##		  a list of valid reserved words(metadata info)
## Output: dictionay of tables and corresponding faulty columns
#############################################################################################
def reserved_word_chk(table_col_dict,reserved_word_lst):
    keylist = table_col_dict.keys()
    
    table_col_resrv_wrd_exist_dict = {}
    
    for ky in keylist:
        tablename = ky
        keyval = table_col_dict.get(ky)
        syntax_chk_flag = keyval["SYNTAX_CHK"]
        if(syntax_chk_flag == "Pass"):
            columnlist = keyval["COLUMN_DESC"].keys()
        ######print tablename
        
            resrv_wrd_exist_col_list = [] 
            
            for column in columnlist:
            	##print reserved_word_lst
            	##print column
                if(column.upper().strip('"') in reserved_word_lst):
                    resrv_wrd_exist_col_list.append(column)
                    
            table_col_resrv_wrd_exist_dict.update({tablename:resrv_wrd_exist_col_list})
        
    return table_col_resrv_wrd_exist_dict
    
#############################################################################################################################################            
## Funtion Name: col_datatype_chk
## Functionality: check if a classword exists in a column name 
##				  then checks if a standard datatype and size is defined against the column
## Input: output dictionary of table_tokenizer function,
##		  a dictionary of classwords,datatype and allowable data length(metadata info)
## Output: dictionay of tables and corresponding columns which arn't having proper datatype and exceeded size
###############################################################################################################
def col_datatype_chk(table_col_dict,col_datatypesz_dict):
    keylist = table_col_dict.keys()
    
    collist = col_datatypesz_dict.keys()
    
    ##refined_col_dict = {}
    ##for colexp in collist
    ##    col = colexp.split('#')
    ##    refined_col_dict.update({colexp:col})
        
    ######print refined_col_dict
    
    table_col_excp_dict = {}
    
    for ky in keylist:
        tablename = ky
        keyval = table_col_dict.get(ky)
        syntax_chk_flag = keyval["SYNTAX_CHK"]
        if(syntax_chk_flag == "Pass"):
            columnlist = keyval["COLUMN_DESC"].keys()
        
            datatypeexcpdict = {}
            
            
            for column in columnlist:
                col_wrd_lst = re.split('[^a-zA-Z0-9]', column.upper().strip())
                col_refind_wrd_lst = [x for x in col_wrd_lst if x != '']
                col_part = col_refind_wrd_lst[len(col_refind_wrd_lst) - 1]
                colexcpflag = False
                
                convertedto = ""
                excpmsg = ""
                for colexp in collist:
                    col = colexp.split('#')[0]
                    
                    ##if(tablename == "PERF_APP.ITS_BILLING_SOU170806"):
                    ##    ##print "************************"
                    ##    ##print col_refind_wrd_lst
                    ##    ##print colexcpflag
                    ##    ##print tablename
                    ##    ##print col
                    ##    ##print column
                    ##    ##print col_part
                    ##    ##print colexp
                    ##    ##print keyval["COLUMN_DESC"][column]
                    ##    ##print col_datatypesz_dict[colexp].keys()[0]
                    ##    ##print "************************"
                    if((column.upper() == col.upper() or col_part == col.upper()) and keyval["COLUMN_DESC"][column]["col_data_type"].upper() == col_datatypesz_dict[colexp].keys()[0]):
                        if(keyval["COLUMN_DESC"][column]["col_size"].split(',')[0].isdigit() 
                           and col_datatypesz_dict[colexp].get(col_datatypesz_dict[colexp].keys()[0]).isdigit()
                           and int(keyval["COLUMN_DESC"][column]["col_size"].split(',')[0]) <= int(col_datatypesz_dict[colexp].get(col_datatypesz_dict[colexp].keys()[0]))):
                            colexcpflag = False
                        elif(keyval["COLUMN_DESC"][column]["col_size"].split(',')[0].isdigit() 
                           and col_datatypesz_dict[colexp].get(col_datatypesz_dict[colexp].keys()[0]).isdigit()
                           and int(keyval["COLUMN_DESC"][column]["col_size"].split(',')[0]) > int(col_datatypesz_dict[colexp].get(col_datatypesz_dict[colexp].keys()[0]))):
                            excpmsg = col_datatypesz_dict[colexp].get(col_datatypesz_dict[colexp].keys()[0])
                            ##colexcpflag = True
                            ##if(tablename == "PERF_APP.LOCATION2345912"):
                            ##    ####print "************************"
                            ##    ####print col_refind_wrd_lst
                            ##    ####print colexcpflag
                            ##    ####print tablename
                            ##    ####print col
                            ##    ####print column
                            ##    ####print col_part
                            ##    ####print colexp
                            ##    ####print keyval["COLUMN_DESC"][column]
                            ##    ####print col_datatypesz_dict[colexp].keys()[0]
                            ##    ####print col_datatypesz_dict[colexp].get(col_datatypesz_dict[colexp].keys()[0])
                            ##    ####print keyval["COLUMN_DESC"][column]["col_size"]
                            ##    ####print "************************"
                            colexcpflag = True
                        else:
                            colexcpflag = False    
                        break   
                    elif((column == col.upper() or col_part == col.upper()) and keyval["COLUMN_DESC"][column]["col_data_type"].upper() != col_datatypesz_dict[colexp].keys()[0]):
                        if(convertedto == ""):
                            convertedto = convertedto + col_datatypesz_dict[colexp].keys()[0]
                        else:
                            convertedto = convertedto + "/" + col_datatypesz_dict[colexp].keys()[0]
                        excpmsg = "datatype '" + keyval["COLUMN_DESC"][column]["col_data_type"] + "' should be converted to '" + convertedto + "'"
                        colexcpflag = True
                        ##break
                    ##elif((not (column == col.upper())) and (not (col_part == col.upper()))):
                    ##	colexcpflag = True
                        ##excpmsg=''
                
                if(colexcpflag):
                    datatypeexcpdict.update({column:excpmsg})
                        
            table_col_excp_dict.update({tablename:datatypeexcpdict})
                    
    print table_col_excp_dict                   
    return table_col_excp_dict     
#############################################################################################################################################            
## Funtion Name: starts_ends_anywhr_list_gen
## Functionality: generate lists to get searched in start or anywhere 
##				  or end of a string
## Input: list 
## Output: list of lists
###############################################################################################################
def starts_ends_anywhr_list_gen(iteration_list):
    strats_chk_list = []
    anywhere_chk_list = []
    ends_chk_list = []
	
    for item in iteration_list:
        if((not(item.startswith('%'))) and (item.endswith('%'))):
            strats_chk_list = strats_chk_list + [item.strip().strip('%')]
        elif((item.startswith('%')) and (item.endswith('%'))):
            anywhere_chk_list = anywhere_chk_list + [item.strip().strip('%')]
        elif((item.startswith('%')) and (not(item.endswith('%')))):
            ends_chk_list = ends_chk_list + [item.strip().strip('%')]
            
    return_list = [strats_chk_list,anywhere_chk_list,ends_chk_list]
    
    return return_list
#############################################################################################################################################            
## Funtion Name: transient_stg_chk
## Functionality: check if a tablename contains certain words 
##				  based on that it will return a boolean value
## Input: table name
## Output: boolean
############################################################################################################### 
def transient_stg_chk(tablename,dbname,tbl_pii_exclusn_md_list,stg_db_table_exclusn_md_list):
	
	##tbl_name_wrd_lst = re.split('[^a-zA-Z0-9]', tablename.upper().strip())
	##dbname_wrd_lst = re.split('[^a-zA-Z0-9]', dbname.upper().strip())
	print tablename
	print dbname
	transient_stg_flag = False
	transient_pii_flag = False
	strats_chk_list = []
	anywhere_chk_list = []
	ends_chk_list = []
	
	complete_srch_list = starts_ends_anywhr_list_gen(stg_db_table_exclusn_md_list)
	
	#print complete_srch_list[0]
	#print complete_srch_list[1]
	#print complete_srch_list[2]
	
	if(tablename.upper().startswith(tuple(complete_srch_list[0])) or (any(x in tablename.upper() for x in complete_srch_list[1]))
	   or tablename.upper().endswith(tuple(complete_srch_list[2]))) or (dbname.upper().endswith(tuple(complete_srch_list[2]))):
	    transient_stg_flag = True
	    
	
	complete_srch_list = starts_ends_anywhr_list_gen(tbl_pii_exclusn_md_list)
	
	if(tablename.upper().startswith(tuple(complete_srch_list[0])) or (any(x in tablename.upper() for x in complete_srch_list[1]))
	   or tablename.upper().endswith(tuple(complete_srch_list[2]))):
	    transient_pii_flag = True
	
	print transient_stg_flag
	##print ("table name is: ",tablename.upper())
	##print ("transientpii flag : ",transient_pii_flag)
	
	return_flag_list = [transient_stg_flag,transient_pii_flag]
	
	return return_flag_list
#############################################################################################################################################            
## Funtion Name: table_wise_report
## Functionality: Produce table wise summary and detail repors based on some checks
## Input: output dictionary of table_tokenizer function,
##		  output of the all check functions like
##		  1. class_word_match
##        2. naming_standard_match
##        3. comment_PII_match
##        4. reserved_word_chk
##        5. col_datatype_chk
##        list of tables having exception
## Output: dictionay of complete summary and detail reports of all tables in a single run
###############################################################################################################
def table_wise_report(table_col_dict,naming_excp_table_dict_param,pii_excp_table_dict_param,col_resrv_wrd_exist_dict_param,
                      classwrd_dttp_excp_dict_param,claaswrd_dict_param,excptntbllist,rule_set_status_dict,rule_set_cmnt_dict,
                      tbl_pii_exclusn_md_list,stg_db_table_exclusn_md_list):
    
    tablewisereportdict = {}
    summaryreportlst = []
    detailreportlst = []
    final_status = "Pass"
    error_msg = ""
    
    for key in table_col_dict:
        syntax_chk_flag = table_col_dict[key]["SYNTAX_CHK"]
        if(len(key.split(".")) > 1):
            actual_tbl_name = key.split(".")[1]
        else:
            actual_tbl_name = key
        
        transient_chk_flag_list = transient_stg_chk(actual_tbl_name,table_col_dict[key]["DBNAME"],tbl_pii_exclusn_md_list,stg_db_table_exclusn_md_list)
        if(syntax_chk_flag == "Pass"):
            col_desc_dict = table_col_dict[key]["COLUMN_DESC"]
            keylist_naming = naming_excp_table_dict_param[key].keys()
            keylist_PII = [colname for colname, comment in pii_excp_table_dict_param[key].items() if comment in ["NO_COMMENT","NO_PII"]]
            keylist_datatype = classwrd_dttp_excp_dict_param[key].keys()
            keylist_classword = claaswrd_dict_param[key].keys()
            collist_resrvwrd = col_resrv_wrd_exist_dict_param[key]
            
            naming_std_chk=""
            pii_chk=""
            datatype_chk=""
            reservedwrd_chk=""
            pi_exst_chk=""
            set_propts_chk=""
            tbl_snap_chk=""
            indentity_col_exst_chk =""
            DDLStandardResultValue=""
            commentString=""
            ##tablepropertiesvalue=""
            
            ##print col_desc_dict.keys()
            
            if((len(keylist_naming) <= 0) and (len(keylist_classword) == len(col_desc_dict.keys()))) or (transient_chk_flag_list[0]):
                naming_std_chk = "N0"
            else:
                naming_std_chk = "N1"
                
            if(transient_chk_flag_list[1]) or (transient_chk_flag_list[0]):
                pii_chk = "PII2"
            elif(len(keylist_PII) <= 0):
                pii_chk = "PII0"
            else:
                pii_chk = "PII1"
                
            if(len(keylist_datatype) <= 0) or (transient_chk_flag_list[0]):
                datatype_chk = "D0"
            else:
                datatype_chk = "D1"
                
            if(len(collist_resrvwrd) <= 0) or (transient_chk_flag_list[0]):
                reservedwrd_chk = "R0"
            else:
                reservedwrd_chk = "R1"
                
            if(table_col_dict[key]["SET_PROPERTIES"] == "MULTISET") or (transient_chk_flag_list[0]):
                set_propts_chk = "S0"
            else:
                set_propts_chk = "S1"
                
            if(table_col_dict[key]["PRIMARY_INDEX"].strip() != "") or (transient_chk_flag_list[0]):
                pi_exst_chk = "P0"
            else:
                pi_exst_chk = "P1"
                
            if(table_col_dict[key]["TABLE_SNAP_PROPERTIES"] != "N") or (transient_chk_flag_list[0]):
                tbl_snap_chk = "TS0"
            else:
                tbl_snap_chk = "TS1"
                
            if(table_col_dict[key]["IDENTITY_FLAG"] == "N") or (transient_chk_flag_list[0]):
                indentity_col_exst_chk = "I0"
            else:
                indentity_col_exst_chk = "I1"
                
            table_properties_chk = set_propts_chk + '-' + pi_exst_chk + '-' + tbl_snap_chk + '-' + indentity_col_exst_chk
            table_final_status =  rule_set_status_dict[table_properties_chk] + '-' + naming_std_chk + '-' + datatype_chk + '-' + reservedwrd_chk + '-' + pii_chk
            
            ##print table_properties_chk
            print table_final_status
            ##print rule_set_status_dict
            ##print rule_set_cmnt_dict
            
            DDLStandardResultValue = rule_set_status_dict[table_final_status]
            table_properties_cmnt_string = rule_set_cmnt_dict[table_properties_chk]
            commentString = table_properties_cmnt_string + rule_set_cmnt_dict[table_final_status]
            
            if(key.upper() in excptntbllist and rule_set_status_dict[reservedwrd_chk] != "Fail"):
                DDLStandardResultValue = "RS1"
                commentString = ""
                    
            commentString = commentString + "See Details in column level review results"
            print DDLStandardResultValue
            print commentString
                    
            
            if(len(key.split(".")) > 1):
                actual_tbl_name = key.split(".")[1]
            else:
                actual_tbl_name = key
             
            concatflag=False 
            refined_partition_list=[]  
            for word in table_col_dict[key]["PARTITION_COLS"]:
                if(word.startswith('"') and not word.endswith('"')):
                    current_word=""
                    concatflag=True
                elif(not concatflag):
                    current_word=word
                
                if(concatflag):
                    current_word = current_word + " " + word
                    if(word.endswith('"')):
                        concatflag=False
                if(not concatflag):
                    refined_partition_list = refined_partition_list + [current_word.strip()]
            ####print refined_partition_list
                    
            col_revw_details_list = []
            partition_colmns=""
            for col in  col_desc_dict:
                col_comment = ""
                column_name = col
                column_datatype = col_desc_dict[col]["col_data_type"]
                column_size = ""
                
                ##print col
                
                if column_name in refined_partition_list:
                    partition_colmns = partition_colmns + "," + column_name
                
                if(col_desc_dict[col]["col_size"].split(',')[0].isdigit()):
                    column_size = col_desc_dict[col]["col_size"]
                
                allowable_length = ""
                datatypevalue = "D0"
                if(col in keylist_datatype):
                    col_dttp_comment = classwrd_dttp_excp_dict_param[key][col]
                    if(col_dttp_comment.isdigit()):
                        allowable_length = col_dttp_comment
                        datatypevalue = "D1"
                    else:
                        ##col_comment = col_comment + "Does not follow standard data type. For standard data types, please visit this link: https://connectme.apple.com/docs/DOC-900851. "  + col_dttp_comment + "."
                        datatypevalue = "D1"
                        col_comment = col_comment + rule_set_cmnt_dict[datatypevalue] + col_dttp_comment + "."
                
                if(allowable_length != ""):
                    if(int(allowable_length) < int(column_size.split(',')[0])):
                        col_comment = col_comment + rule_set_cmnt_dict[datatypevalue] + "Column size should be with in " + allowable_length + " ."
                        
                namingStandard = "N0"
                if(col in keylist_naming):
                    col_nms_comment = naming_excp_table_dict_param[key][col]
                    namingStandard = "N1"
                    col_comment = col_comment + rule_set_cmnt_dict[namingStandard] + col_nms_comment + "."
                    ##namingStandard = "Warning"
                    
                classword_chk = "C1"
                classword_txt = ""
                if(col in keylist_classword):
                    classword_chk = "C0"
                    classword_txt = claaswrd_dict_param[key][col]
                else:
                    namingStandard = "N1"
                    
                pii_text = ""
                if(pii_excp_table_dict_param[key][col] not in ["NO_COMMENT","NO_PII"]):
                    pii_text = pii_excp_table_dict_param[key][col]
                    
                pii_value = "PII0"
                if(transient_chk_flag_list[1]) or (transient_chk_flag_list[0]):
                    pii_value = "PII2"
                elif(col in keylist_PII):
                    pii_value = "PII1"
                    col_comment = col_comment + rule_set_cmnt_dict[pii_value]
                    
                    
                reserved_wrd_value = "R0"
                if(col in collist_resrvwrd):
                    reserved_wrd_value = "R1"
                    
                col_revw_details_list = col_revw_details_list + [{
                                                                  "DatabaseName": table_col_dict[key]["DBNAME"],
                                                                  "NamingStandardValue": rule_set_status_dict[namingStandard],
                                                                  "ColumnDatatypeValue": column_datatype,
                                                                  "CommentText": col_comment,
                                                                  "DatatypeValue": rule_set_status_dict[datatypevalue],
                                                                  "PIIText": pii_text,
                                                                  "ColumnClassCode": classword_txt,
                                                                  "ReservedWordValue": rule_set_status_dict[reserved_wrd_value],
                                                                  "fileName": table_col_dict[key]["FILENAME"],
                                                                  "ColumnLengthValue": column_size,
                                                                  "TableName": actual_tbl_name,
                                                                  "AllowableLengthValue": allowable_length,
                                                                  "ColumnName": col,
                                                                  "ClassWordValue": rule_set_status_dict[classword_chk],
                                                                  "PIIValue": rule_set_status_dict[pii_value],
                                                                  "Compressed":table_col_dict[key]["COLUMN_DESC"][col]["compress_flag"],
                                                                  "Title":table_col_dict[key]["COLUMN_DESC"][col]["title"],
                                                                  "IdentityColumnValue": table_col_dict[key]["COLUMN_DESC"][col]["identity_flag"]
                                                                }]
            ##tablewisedetailsdict.update({key:col_revw_details_dict})
            summaryreportlst = summaryreportlst + [{"DatabaseName": table_col_dict[key]["DBNAME"],
                                                    "NamingStandardValue": rule_set_status_dict[naming_std_chk],
                                                    "MultisetValue": rule_set_status_dict[set_propts_chk],
                                                    "ReservedWordValue": rule_set_status_dict[reservedwrd_chk],
                                                    "fileName": table_col_dict[key]["FILENAME"],
                                                    "NoPIValue": rule_set_status_dict[pi_exst_chk],
                                                    "TableName": actual_tbl_name,
                                                    "DDLStandardResultValue": rule_set_status_dict[DDLStandardResultValue],
                                                    "TablePropertiesValue": rule_set_status_dict[tbl_snap_chk],
                                                    "PIIValue": rule_set_status_dict[pii_chk],
                                                    "IdentityColumnValue": rule_set_status_dict[indentity_col_exst_chk],
                                                    "CommentText": commentString,
                                                    "DatatypeValue": rule_set_status_dict[datatype_chk],
                                                    "Buckets":'NA',
                                                    "PartitioningColumn":partition_colmns.strip(","),
                                                    "BucketedColumn":'NA',
                                                    "PrimaryKey":table_col_dict[key]["PRIMARY_INDEX"]}]
                                                
            detailreportlst = detailreportlst + col_revw_details_list
            if(rule_set_status_dict[DDLStandardResultValue] == "Fail"):
                error_msg = error_msg + "The table " + key + " failed for " + commentString + " "
                
            if(final_status != "Fail" and rule_set_status_dict[DDLStandardResultValue] in ["Fail","Warning"]):
                final_status = rule_set_status_dict[DDLStandardResultValue]
                ####print final_status
                
        else:
            
            if(len(key.split(".")) > 1):
                actual_tbl_name = key.split(".")[1]
            else:
                actual_tbl_name = key
                
            summaryreportlst = summaryreportlst + [{"DatabaseName": table_col_dict[key]["DBNAME"],
                                                    "NamingStandardValue": "Fail",
                                                    "MultisetValue": "Fail",
                                                    "ReservedWordValue": "Fail",
                                                    "fileName": table_col_dict[key]["FILENAME"],
                                                    "NoPIValue": "Fail",
                                                    "TableName": actual_tbl_name,
                                                    "DDLStandardResultValue": "Fail",
                                                    "TablePropertiesValue": "Fail",
                                                    "PIIValue": "Fail",
                                                    "IdentityColumnValue": "Fail",
                                                    "CommentText": "syntax is not correct",
                                                    "DatatypeValue": "Fail",
                                                    "PrimaryKey":None}]
                                                    
            error_msg = error_msg + "The table " + key + " syntax is not correct. "
            final_status = "Fail"                                        
            ##tablewisedetailsdict = {}
            
    tablewisereportdict.update({"status": final_status,
                                "summarizedResult": summaryreportlst,
                                "detailedResult":detailreportlst,
                                "errorMessage":error_msg})
                                
    ##returnlist = [tablewisesummarydict,tablewisedetailsdict]    
                
    return  tablewisereportdict    
######################################################**Main Execution Starts Here**##########################################################
def reportmain(path, inputjson):
    mainexecfilepath = path  #####taking the full path of the main executable file from command line argument 
    
    fileDir = os.path.dirname(os.path.realpath(mainexecfilepath)) ####storing the directory path of the main file
    print fileDir
    
    parentfileDir = os.path.dirname(fileDir)
    print parentfileDir
    
    error_msg = ""
    
    log_dir_path = parentfileDir + "/Pythonlog"  ###Creating a log folder if not exists
    ensure_dir(log_dir_path)
    
    metadata_dir = parentfileDir + "/Metadata"
    
    DM_CLASSWORD = metadata_dir + "/DM_CLASSWORD.txt"
    
    with open(DM_CLASSWORD,'r') as infile:
        class_word_lis = [line.strip() for line in infile]  ###reading the class words from a file and storing in a list
        
    
    DM_RESERVED_WORD = metadata_dir + "/DM_RESERVED_WORD.txt"
    
    with open(DM_RESERVED_WORD,'r') as infile:
        reserved_word_lis = [line.upper().strip() for line in infile] ###reading the reserved words from a file and storing in a list
    
    
    DM_TABLE_EXCEPTION = metadata_dir + "/DM_TABLE_EXCEPTION.txt"
    
    with open(DM_TABLE_EXCEPTION,'r') as infile:
        dm_table_exception_lis = [line.upper().strip() for line in infile] ###reading the exceptional table names from a file and storing in a list
    
    DM_NAMING_STD = metadata_dir + "/DM_NAMING_STD.csv"
    
    with open(DM_NAMING_STD, mode='r') as infile:
        reader = csv.reader(infile)
        namingcnvdict = {}
        for rows in reader: 
            namingcnvdict.update({rows[0].upper():rows[1].upper()}) ###reading from a csv file and storing in a dictionary
    
    
    DM_CLASS_DATATYPE = metadata_dir + "/DM_CLASS_DATATYPE.csv"
        
    with open(DM_CLASS_DATATYPE, mode='r') as infile:
        reader = csv.reader(infile)
        cnt=0
        coldttpszdict = {}
        for rows in reader:
          coldttpszdict.update({(rows[0].upper() + "#" + str(cnt)):{rows[1].upper():rows[2]}}) ###reading from a csv file and storing in a dictionary. concatenating cnt to generate uniq key
          cnt = cnt + 1
          
          
    PII_CLASSIFICATION_LIST = metadata_dir + "/PII_CLASSIFICATION_LIST.txt"
    
    with open(PII_CLASSIFICATION_LIST,'r') as infile:
        PII_wrd_lis = [line.strip() for line in infile]      
    
    TBL_PII_EXCLUSN_MD_LIST = metadata_dir + "/tbl_pii_exclusn_md_list.txt"
    
    with open(TBL_PII_EXCLUSN_MD_LIST,'r') as infile:
        tbl_pii_exclusn_md_list = [line.strip().upper() for line in infile]
        
    STG_DB_TABLE_EXCLUSN_MD_LIST = metadata_dir + "/stg_db_table_exclusn_md_list.txt"
    
    with open(STG_DB_TABLE_EXCLUSN_MD_LIST,'r') as infile:
        stg_db_table_exclusn_md_list = [line.strip().upper() for line in infile]
    
    Rule_set = metadata_dir + "/rule_md.csv"
    
    csv.register_dialect('piper', delimiter='|', quoting=csv.QUOTE_NONE)
    with open(Rule_set, mode='r') as infile:
        reader = csv.reader(infile, dialect='piper')
        
        rule_set_status_dict = {}
        rule_set_cmnt_dict = {}
        for rows in reader:
            rule_set_status_dict.update({rows[0]:rows[1]})
            rule_set_cmnt_dict.update({rows[0]:rows[2]})
    
    ddldir = parentfileDir + "/ddl"
    ensure_dir(ddldir)
    
    prsnttime = datetime.datetime.now().strftime ("%Y%m%d%H%M%S")   
    ddldirforinstance = ddldir + "/ddl_" + prsnttime
    ensure_dir(ddldirforinstance)
    
    log_dir_path = log_dir_path + "/log_" + prsnttime
    ensure_dir(log_dir_path)
    
    log_file = log_dir_path + "/DM_script.log"
    tabl_wrd_dtls_file = log_dir_path + "/tabl_wrd_dtls.json"
    
    logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                        format='%(asctime)s: '
                               '%(filename)s: '    
                               '%(levelname)s: '
                               '%(funcName)s(): '
                               '%(lineno)d:\t'
                               '%(message)s')
    logger=logging.getLogger(__name__)
    
    try:
        filesddlsdict = eval(inputjson.replace('\\r','\\\n').replace('\\n','\\\n<dm tool line brk is here>'))
        print filesddlsdict
        filesddlspylst = filesddlsdict["InputValues"]
        print "table ddl is:"
        print filesddlspylst
    	
        try:
            tablewordmap = table_tokenizer(ddldirforinstance,filesddlspylst,coldttpszdict,log_dir_path,fileDir)
        except Exception as e:
            error_msg = error_msg + "::" + str(e)
            print error_msg
            logger.error(e)
            raise Exception("Error occured in table_tokenizer()")
        
        #print "================== TABLE WORD MAP =================="
        #print tablewordmap
        
        if(any(tablewordmap)):
            try:
                col_cmnt_dict = comment_PII_match(tablewordmap,PII_wrd_lis)
            except Exception as e:
                error_msg = error_msg + "::" + str(e)
                print error_msg
                logger.error(e)
                raise Exception("Error occured in comment_PII_match()")
                
            try:
                col_nming_std_excp_dict = naming_standard_match(tablewordmap,namingcnvdict)
            except Exception as e:
                error_msg = error_msg + "::" + str(e)
                print error_msg
                logger.error(e)
                raise Exception("Error occured in naming_standard_match()")
                
            try:
                col_dttp_sz_excp_dict = col_datatype_chk(tablewordmap,coldttpszdict)
            except Exception as e:
                error_msg = error_msg + "::" + str(e)
                print error_msg
                logger.error(e)
                raise Exception("Error occured in col_datatype_chk()")
                
            try:
                col_rsrv_wrd_exst_dict = reserved_word_chk(tablewordmap,reserved_word_lis)
            except Exception as e:
                error_msg = error_msg + "::" + str(e)
                print error_msg
                logger.error(e)
                raise Exception("Error occured in reserved_word_chk()")
                
            try:
                col_cls_wrd_mtch_excp_dict = class_word_match(tablewordmap,class_word_lis)
            except Exception as e:
                error_msg = error_msg + "::" + str(e)
                print error_msg
                logger.error(e)
                raise Exception("Error occured in class_word_match()")
                
            try:
                table_wise_rpt_dict = table_wise_report(tablewordmap,col_nming_std_excp_dict,col_cmnt_dict,col_rsrv_wrd_exst_dict,
                                                            col_dttp_sz_excp_dict,col_cls_wrd_mtch_excp_dict,dm_table_exception_lis,
                                                            rule_set_status_dict,rule_set_cmnt_dict,tbl_pii_exclusn_md_list,stg_db_table_exclusn_md_list)
                shutil.rmtree(ddldirforinstance)
            except Exception as e:
                error_msg = error_msg + "::" + str(e)
                print error_msg
                logger.error(e)
                raise Exception("Error occured in table_wise_report()")
            
			### Removes logs after a successful execution
            shutil.rmtree(log_dir_path)
            finalreportJSON = json.dumps(table_wise_rpt_dict,sort_keys=False, indent=4)
            print "Final Report :"
            print "*****************"
            print finalreportJSON    
            return finalreportJSON      
            
        else:
             raise Exception("There is no table to scan.")      
    except Exception as e:
        with open(tabl_wrd_dtls_file, 'w') as fp:
            json.dump(tablewordmap, fp,sort_keys=True, indent=4)
        shutil.rmtree(ddldirforinstance)
        error_msg = error_msg + "::" + str(e)
        print error_msg
        sys.exit(1)
