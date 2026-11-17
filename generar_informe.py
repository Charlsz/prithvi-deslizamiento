"""Compone el informe PDF desde Markdown y las figuras reales de entregables."""
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent


def main():
    for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('Text',fontName='Arial',fontSize=10,leading=14,spaceAfter=8,splitLongWords=True))
    styles.add(ParagraphStyle('DocTitle',fontName='Arial-Bold',fontSize=24,leading=29,spaceAfter=17,textColor=colors.HexColor('#162c3e')))
    styles.add(ParagraphStyle('Section',fontName='Arial-Bold',fontSize=14,leading=18,spaceBefore=14,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#162c3e')))
    styles.add(ParagraphStyle('Cell',fontName='Arial',fontSize=8,leading=11,spaceAfter=0,splitLongWords=True))
    styles.add(ParagraphStyle('Caption',fontName='Arial',fontSize=8.5,leading=11,spaceAfter=10,textColor=colors.HexColor('#42586b')))
    def markup(s):
        s=escape(s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
        s=s.replace('\n','<br/>')
        return s
    lines=(ROOT/'Grupo_2_Informe.md').read_text(encoding='utf-8').splitlines()
    story=[];i=0; width=483
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):story.append(Paragraph(markup(line[2:]),styles['DocTitle']));i+=1;continue
        if line.startswith('### '):story.append(Paragraph(markup(line[4:]),styles['Section']));i+=1;continue
        if line.startswith('## '):story.append(Paragraph(markup(line[3:]),styles['Section']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                parts=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+',x) for x in parts):rows.append(parts)
                i+=1
            n=len(rows[0])
            if n==2:weights=[.37,.63]
            elif n==3:weights=[.2,.24,.56] if rows[0][0]=='Caso' else [.5,.25,.25]
            elif n==4:weights=[.15,.39,.23,.23]
            elif n==6:weights=[.26,.14,.15,.15,.15,.15]
            else:weights=[1/n]*n
            table=Table([[Paragraph(markup(x),styles['Cell']) for x in row] for row in rows],colWidths=[width*w for w in weights],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9aaab7')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
            story.extend([KeepTogether([table]),Spacer(1,10)]);continue
        match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if match:
            img=Image(str(ROOT/match[2]));img.drawHeight=width*img.imageHeight/img.imageWidth;img.drawWidth=width
            story.append(KeepTogether([img,Paragraph(markup(match[1]),styles['Caption'])]));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
            para.append(lines[i].strip());i+=1
        joiner='\n' if all(x.startswith('**') for x in para) else ' '
        style=styles['Caption'] if line.startswith('[') and re.match(r'\[\d+\]',line) else styles['Text']
        story.append(Paragraph(markup(joiner.join(para)),style))
    pdf=ROOT/'Grupo_2_Informe.pdf'
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Arial',8);canvas.setFillColor(colors.HexColor('#42586b'))
        canvas.drawString(56,29,'Grupo 2  |  Detección de deslizamientos con Prithvi')
        canvas.drawRightString(539,29,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf),pagesize=(595,842),rightMargin=56,leftMargin=56,topMargin=48,bottomMargin=49,title='Detección de deslizamientos con Prithvi',author='Grupo 2')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(pdf)


if __name__=='__main__':main()
