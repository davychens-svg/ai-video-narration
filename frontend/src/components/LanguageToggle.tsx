import React from 'react';
import { Languages } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Language, getTranslations } from '../lib/i18n';

interface LanguageToggleProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
}

export function LanguageToggle({ language, onLanguageChange }: LanguageToggleProps) {
  const t = getTranslations(language);

  return (
    <div className="flex items-center gap-2">
      <Languages className="w-4 h-4 text-muted-foreground" />
      <Select value={language} onValueChange={(value) => onLanguageChange(value as Language)}>
        <SelectTrigger className="w-[140px] h-9 bg-background/50 border-border/50">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="en">{t.languageEnglish}</SelectItem>
          <SelectItem value="ja">{t.languageJapanese}</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
