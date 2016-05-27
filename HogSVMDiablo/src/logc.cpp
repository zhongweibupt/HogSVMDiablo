/*
* logc.c
*
*/
#define _CRT_SECURE_NO_DEPRECATE

#include "logc.h"
#define MAXLEVELNUM (3)
#define CONFIG_PATH  "C://Users//zhwei//Desktop//Diablo//logs//log.conf"
//#define LEVEL ALL
LOGSET logsetting;
LOG loging;

const static char LogLevelText[4][10] = { "INF", "DEBUG", "ERROR", "ERROR" };

static char * getdate(char *date);

static unsigned char getcode(char *path){
	unsigned char code = 255;
	if (strcmp("INF", path) == 0)
		code = 1;
	else if (strcmp("DEBUG", path) == 0)
		code = 2;
	else if (strcmp("ERROR", path) == 0)
		code = 4;
	else if (strcmp("NONE", path) == 0)
		code = 0;
	return code;
}

static unsigned char ReadConfig(char *path){
	char value[512] = { 0x0 };
	char data[50] = { 0x0 };

	FILE *fpath = fopen(path, "r");
	if (fpath == NULL)
		return -1;
	fscanf(fpath, "path=%s\n", value);
	getdate(data);
	strcat(data, ".log");
	strcat(value, "//");
	strcat(value, data);
	if (strcmp(value, logsetting.filepath) != 0)
		memcpy(logsetting.filepath, value, strlen(value));
	memset(value, 0, sizeof(value));

	fscanf(fpath, "level=%s\n", value);
	logsetting.loglevel = getcode(value);
	fclose(fpath);
	return 0;
}
/*
*��־������Ϣ
* */
static LOGSET *getlogset(){
	char path[512] = { 0x0 };
	//getcwd(path, sizeof(path));
	strcat(path, CONFIG_PATH);
	//if (access(path, F_OK) == 0){
		if (ReadConfig(path) != 0){
			logsetting.loglevel = INF;
			logsetting.maxfilelen = 4096;
		}
	//}
	//else{
		//logsetting.loglevel = INF;
		//logsetting.maxfilelen = 4096;
	//}
	return &logsetting;
}

/*
*��ȡ����
* */
static char * getdate(char *date){
	time_t timer = time(NULL);
	strftime(date, 11, "%Y-%m-%d", localtime(&timer));
	return date;
}

/*
*��ȡʱ��
* */
static void settime(){
	time_t timer = time(NULL);
	strftime(loging.logtime, 20, "%Y-%m-%d %H:%M:%S", localtime(&timer));
}

/*
*�����δ�ӡ
* */
static void PrintfLog(char * fromat, va_list args){
	int d;
	char c, *s;
	while (*fromat)
	{
		switch (*fromat){
		case 's':{
			s = va_arg(args, char *);
			fprintf(loging.logfile, "%s", s);
			break; }
		case 'd':{
			d = va_arg(args, int);
			fprintf(loging.logfile, "%d", d);
			break; }
		case 'c':{
			c = (char)va_arg(args, int);
			fprintf(loging.logfile, "%c", c);
			break; }
		default:{
			if (*fromat != '%'&&*fromat != '\n')
				fprintf(loging.logfile, "%c", *fromat);
			break; }
		}
		fromat++;
	}
	fprintf(loging.logfile, "%s", "]\n");
}

static int initlog(unsigned char loglevel){
	char strdate[30] = { 0x0 };
	LOGSET *logsetting;
	//��ȡ��־������Ϣ
	if ((logsetting = getlogset()) == NULL){
		perror("Get Log Set Fail!");
		return -1;
	}
	//strcpy(logsetting->filepath, PATH);
	//logsetting->loglevel = LEVEL;

	if ((loglevel&(logsetting->loglevel)) != loglevel)
		return -1;

	memset(&loging, 0, sizeof(LOG));
	//��ȡ��־ʱ��
	settime();
	if (strlen(logsetting->filepath) == 0){
		char *path = getenv("HOME");
		memcpy(logsetting->filepath, path, strlen(path));

		getdate(strdate);
		strcat(strdate, ".log");
		strcat(logsetting->filepath, "/");
		strcat(logsetting->filepath, strdate);
	}
	memcpy(loging.filepath, logsetting->filepath, MAXFILEPATH);
	//����־�ļ�
	if (loging.logfile == NULL)
		loging.logfile = fopen(loging.filepath, "a+");
	if (loging.logfile == NULL){
		perror("Open Log File Fail!");
		return -1;
	}
	//д����־������־ʱ��
	fprintf(loging.logfile, "[%s] [%s]:[", LogLevelText[loglevel - 1], loging.logtime);
	return 0;
}

/*
*��־д��
* */
int LogWrite(unsigned char loglevel, char *fromat, ...){
	va_list args;
	//��ʼ����־
	if (initlog(loglevel) != 0)
		return -1;
	//��ӡ��־��Ϣ
	va_start(args, fromat);
	PrintfLog(fromat, args);
	va_end(args);
	//�ļ�ˢ��
	fflush(loging.logfile);
	//��־�ر�
	if (loging.logfile != NULL)
		fclose(loging.logfile);
	loging.logfile = NULL;
	return 0;
}